# debug_column_mapping.py - Find the exact column mapping problem
import psycopg2
import json

def debug_column_mapping():
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research", 
        user="researcher",
        password="academic_password_2024"
    )
    cur = conn.cursor()
    
    print("🔍 COLUMN MAPPING DIAGNOSIS:")
    print("=" * 60)
    
    # 1. Check exact table structure with ordinal positions
    print("1. EXACT TABLE STRUCTURE (WITH POSITIONS):")
    cur.execute("""
        SELECT ordinal_position, column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = 'kmeans_segments' 
        AND table_schema = 'casino_data'
        ORDER BY ordinal_position;
    """)
    
    columns = cur.fetchall()
    for pos, col_name, data_type, nullable in columns:
        print(f"   Position {pos:2d}: {col_name:25} | {data_type:15} | Nullable: {nullable}")
    
    # 2. Test with EXPLICIT column names to see what happens
    print("\n2. TESTING EXPLICIT INSERT:")
    
    test_data = {
        "customer_id": 88888,
        "period_id": "2022-H1", 
        "kmeans_version": 1,
        "cluster_id": 2,
        "cluster_label": "Test_Debug_Player",
        "silhouette_score": 0.75,
        "distance_to_centroid": 1.23,
        "model_metadata": {"test": "metadata", "score": 0.75},
        "avg_session_from_metadata": 9999.99,
        "segment_data": {"test": "segment", "label": "Test_Debug_Player"}
    }
    
    # Test explicit insert with all column names
    try:
        cur.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, kmeans_version, cluster_id, cluster_label,
                silhouette_score, distance_to_centroid, model_metadata,
                avg_session_from_metadata, segment_data
            )
            VALUES (%(customer_id)s, %(period_id)s, %(kmeans_version)s, %(cluster_id)s, %(cluster_label)s,
                    %(silhouette_score)s, %(distance_to_centroid)s, %(model_metadata)s::jsonb,
                    %(avg_session_from_metadata)s, %(segment_data)s::jsonb)
        """, {
            "customer_id": test_data["customer_id"],
            "period_id": test_data["period_id"],
            "kmeans_version": test_data["kmeans_version"], 
            "cluster_id": test_data["cluster_id"],
            "cluster_label": test_data["cluster_label"],
            "silhouette_score": test_data["silhouette_score"],
            "distance_to_centroid": test_data["distance_to_centroid"],
            "model_metadata": json.dumps(test_data["model_metadata"]),
            "avg_session_from_metadata": test_data["avg_session_from_metadata"],
            "segment_data": json.dumps(test_data["segment_data"])
        })
        
        print("   ✅ Explicit INSERT successful!")
        
        # Check what was actually inserted
        cur.execute("""
            SELECT customer_id, cluster_label, avg_session_from_metadata,
                   model_metadata, segment_data
            FROM casino_data.kmeans_segments 
            WHERE customer_id = %s
        """, (test_data["customer_id"],))
        
        result = cur.fetchone()
        if result:
            print(f"   📊 Retrieved: Customer {result[0]} | Label: {result[1]} | AvgSession: {result[2]}")
            print(f"   📊 Metadata: {result[3]}")
            print(f"   📊 Segment: {result[4]}")
        
    except Exception as e:
        print(f"   ❌ Explicit INSERT failed: {e}")
    
    # 3. Check your current Python code's INSERT statement format
    print("\n3. CHECKING CURRENT PYTHON INSERT FORMAT:")
    print("   Your Python code uses this INSERT:")
    print("""
        INSERT INTO casino_data.kmeans_segments (
            customer_id, period_id, kmeans_version, cluster_id, cluster_label,
            silhouette_score, distance_to_centroid,
            model_metadata, avg_session_from_metadata, segment_data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
    """)
    
    # 4. Test the exact same format your Python uses
    test_tuple = (
        77777, "2022-H1", 1, 1, "Python_Format_Test",
        0.80, 0.5, json.dumps({"python": "test"}), 7777.77, json.dumps({"format": "test"})
    )
    
    try:
        cur.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, kmeans_version, cluster_id, cluster_label,
                silhouette_score, distance_to_centroid,
                model_metadata, avg_session_from_metadata, segment_data
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
        """, test_tuple)
        
        print("   ✅ Python format INSERT successful!")
        
        # Check result
        cur.execute("""
            SELECT customer_id, cluster_label, avg_session_from_metadata,
                   model_metadata->>'python' as metadata_test,
                   segment_data->>'format' as segment_test
            FROM casino_data.kmeans_segments 
            WHERE customer_id = %s
        """, (77777,))
        
        result = cur.fetchone()
        if result:
            print(f"   📊 Python Format Result:")
            print(f"      Customer: {result[0]} | Label: {result[1]} | AvgSession: {result[2]}")
            print(f"      Metadata Test: {result[3]} | Segment Test: {result[4]}")
            
            if result[2] is not None and result[3] is not None:
                print("   🎉 PYTHON FORMAT WORKS CORRECTLY!")
            else:
                print("   ❌ PYTHON FORMAT STILL FAILING!")
        
    except Exception as e:
        print(f"   ❌ Python format INSERT failed: {e}")
    
    # 5. Clean up test records
    cur.execute("DELETE FROM casino_data.kmeans_segments WHERE customer_id IN (88888, 77777)")
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    debug_column_mapping()