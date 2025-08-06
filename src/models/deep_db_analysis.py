# deep_db_analysis.py - Find the REAL problem with NULL values
import psycopg2
import json

def deep_database_analysis():
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research",
        user="researcher", 
        password="academic_password_2024"
    )
    cur = conn.cursor()
    
    print("🕵️ DEEP DATABASE ANALYSIS:")
    print("=" * 70)
    
    # 1. Check if we're looking at the RIGHT table/records
    print("1. RECENT RECORDS CHECK:")
    cur.execute("""
        SELECT customer_id, period_id, cluster_label, created_at,
               avg_session_from_metadata, segment_data,
               model_metadata IS NOT NULL as has_metadata
        FROM casino_data.kmeans_segments 
        WHERE period_id = '2022-H1'
        ORDER BY created_at DESC 
        LIMIT 5;
    """)
    
    print("   Most Recent Records:")
    for row in cur.fetchall():
        print(f"   Customer: {row[0]} | Period: {row[1]} | Label: {row[2]} | Time: {row[3]}")
        # FIXED: Safe string slicing for NULL values
        segment_preview = str(row[5])[:50] if row[5] is not None else 'NULL'
        print(f"   AvgSession: {row[4]} | SegmentData: {segment_preview} | HasMeta: {row[6]}")
        print()
    
    # 2. Check if the INSERT is actually creating NEW records or updating OLD ones
    print("2. CONFLICT RESOLUTION CHECK:")
    cur.execute("""
        SELECT customer_id, period_id, kmeans_version, created_at,
               avg_session_from_metadata IS NOT NULL as has_avg_session,
               segment_data IS NOT NULL as has_segment_data
        FROM casino_data.kmeans_segments 
        WHERE period_id = '2022-H1' 
        AND customer_id = 255319  -- From your debug log
        ORDER BY created_at DESC;
    """)
    
    print("   Customer 255319 Records:")
    for row in cur.fetchall():
        print(f"   ID: {row[0]} | Period: {row[1]} | Version: {row[2]} | Time: {row[3]}")
        print(f"   HasAvgSession: {row[4]} | HasSegmentData: {row[5]}")
    
    # 3. Test DIRECT INSERT without ON CONFLICT to see if that works
    print("\n3. TESTING DIRECT INSERT (NO CONFLICT RESOLUTION):")
    
    try:
        cur.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, cluster_id, cluster_label,
                silhouette_score, distance_to_centroid, model_metadata,
                kmeans_version, segment_data, avg_session_from_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s)
        """, (
            99999, '2022-H1', 2, 'Direct_Test_Player',
            0.85, 1.5, json.dumps({"direct": "test", "value": 123}),
            1, json.dumps({"segment": "direct_test", "label": "Direct_Test_Player"}), 8888.88
        ))
        
        print("   ✅ Direct INSERT successful!")
        
        # Check what was inserted
        cur.execute("""
            SELECT customer_id, cluster_label, avg_session_from_metadata,
                   segment_data->>'label' as segment_label,
                   model_metadata->>'direct' as metadata_test
            FROM casino_data.kmeans_segments 
            WHERE customer_id = 99999
        """)
        
        result = cur.fetchone()
        if result:
            print(f"   📊 Direct Result: Customer {result[0]} | Label: {result[1]}")
            print(f"   📊 AvgSession: {result[2]} | SegmentLabel: {result[3]} | MetaTest: {result[4]}")
            
            if result[2] is not None and result[3] is not None:
                print("   🎉 DIRECT INSERT WORKS! Problem is ON CONFLICT clause!")
            else:
                print("   ❌ Even direct insert fails - deeper schema issue!")
    
    except Exception as e:
        print(f"   ❌ Direct INSERT failed: {e}")
    
    # 4. Check constraint/trigger issues
    print("\n4. CHECKING CONSTRAINTS & TRIGGERS:")
    cur.execute("""
        SELECT conname, contype, pg_get_constraintdef(oid) as definition
        FROM pg_constraint
        WHERE conrelid = 'casino_data.kmeans_segments'::regclass;
    """)
    
    print("   Table Constraints:")
    for row in cur.fetchall():
        print(f"   {row[0]} ({row[1]}): {row[2]}")
    
    # Check triggers
    cur.execute("""
        SELECT tgname, tgtype, prosrc 
        FROM pg_trigger t
        JOIN pg_proc p ON t.tgfoid = p.oid
        WHERE tgrelid = 'casino_data.kmeans_segments'::regclass;
    """)
    
    triggers = cur.fetchall()
    if triggers:
        print("   Table Triggers:")
        for row in triggers:
            print(f"   {row[0]}: {row[1]}")
    else:
        print("   No triggers found.")
    
    # 5. Check if columns are being OVERWRITTEN by defaults/triggers
    print("\n5. CHECKING COLUMN DEFAULTS:")
    cur.execute("""
        SELECT column_name, column_default, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'kmeans_segments' 
        AND table_schema = 'casino_data'
        AND column_name IN ('avg_session_from_metadata', 'segment_data', 'model_metadata')
        ORDER BY ordinal_position;
    """)
    
    print("   Column Defaults:")
    for row in cur.fetchall():
        print(f"   {row[0]}: default={row[1]}, nullable={row[2]}")
    
    # 6. Test ON CONFLICT behavior specifically
    print("\n6. TESTING ON CONFLICT BEHAVIOR:")
    
    # First, insert a base record
    try:
        cur.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, kmeans_version, cluster_id, cluster_label
            )
            VALUES (88888, '2022-H1', 1, 0, 'Base_Record')
        """)
        
        # Now test the ON CONFLICT UPDATE
        cur.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, cluster_id, cluster_label,
                silhouette_score, distance_to_centroid, model_metadata,
                kmeans_version, segment_data, avg_session_from_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s)
            ON CONFLICT (customer_id, period_id, kmeans_version) DO UPDATE
            SET cluster_id = EXCLUDED.cluster_id,
                cluster_label = EXCLUDED.cluster_label,
                avg_session_from_metadata = EXCLUDED.avg_session_from_metadata,
                segment_data = EXCLUDED.segment_data
        """, (
            88888, '2022-H1', 1, 'Conflict_Test_Player',
            0.90, 2.0, json.dumps({"conflict": "test"}),
            1, json.dumps({"conflict_segment": "test"}), 6666.66
        ))
        
        # Check result
        cur.execute("""
            SELECT cluster_label, avg_session_from_metadata,
                   segment_data->>'conflict_segment' as segment_test
            FROM casino_data.kmeans_segments 
            WHERE customer_id = 88888
        """)
        
        result = cur.fetchone()
        print(f"   ON CONFLICT Result: Label={result[0]}, AvgSession={result[1]}, SegmentTest={result[2]}")
        
        if result[1] is not None and result[2] is not None:
            print("   🎉 ON CONFLICT works correctly!")
        else:
            print("   ❌ ON CONFLICT is causing the NULL issue!")
            
    except Exception as e:
        print(f"   ❌ ON CONFLICT test failed: {e}")
    
    # Clean up test records
    cur.execute("DELETE FROM casino_data.kmeans_segments WHERE customer_id IN (99999, 88888)")
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    deep_database_analysis()