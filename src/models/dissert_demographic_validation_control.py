"""
Data Validation Check
Verify nationality data accuracy and check for missing Turkish customers
"""

import psycopg2
import pandas as pd

DB_CONFIG = {
    'host': 'localhost',
    'database': 'casino_research',
    'user': 'researcher',
    'password': 'academic_password_2024'
}

def validate_nationality_data():
    """Validate nationality data and check for discrepancies."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("DATA VALIDATION CHECK")
        print("=" * 30)
        
        # 1. Check total customers in demographic table
        print("1. Demographics table validation:")
        cursor.execute("SELECT COUNT(*) FROM casino_data.customer_demographics")
        total_demo = cursor.fetchone()[0]
        print(f"   Total customers in demographics: {total_demo:,}")
        
        # 2. Check nationality distribution in demographics
        cursor.execute("""
            SELECT nationality, COUNT(*) as count
            FROM casino_data.customer_demographics 
            WHERE nationality IS NOT NULL
            GROUP BY nationality 
            ORDER BY count DESC
            LIMIT 15
        """)
        
        demo_nationalities = cursor.fetchall()
        print(f"\n   Top 15 nationalities in demographics:")
        for nationality, count in demo_nationalities:
            print(f"     {nationality}: {count:,}")
        
        # 3. Check if Turkey exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM casino_data.customer_demographics 
            WHERE nationality ILIKE '%turkey%' OR nationality ILIKE '%turkish%'
        """)
        turkey_count = cursor.fetchone()[0]
        print(f"\n   Turkey/Turkish customers: {turkey_count:,}")
        
        # 4. Check promo_label table
        print(f"\n2. Promo label table validation:")
        cursor.execute("SELECT COUNT(*) FROM casino_data.promo_label")
        total_promo = cursor.fetchone()[0]
        print(f"   Total records in promo_label: {total_promo:,}")
        
        cursor.execute("SELECT COUNT(DISTINCT customer_id) FROM casino_data.promo_label")
        unique_promo = cursor.fetchone()[0]
        print(f"   Unique customers in promo_label: {unique_promo:,}")
        
        # 5. Check JOIN coverage
        print(f"\n3. JOIN validation:")
        cursor.execute("""
            SELECT COUNT(*)
            FROM casino_data.customer_demographics cd
            INNER JOIN casino_data.promo_label pl ON cd.customer_id = pl.customer_id
        """)
        joined_records = cursor.fetchone()[0]
        print(f"   Records after JOIN: {joined_records:,}")
        
        cursor.execute("""
            SELECT COUNT(DISTINCT cd.customer_id)
            FROM casino_data.customer_demographics cd
            INNER JOIN casino_data.promo_label pl ON cd.customer_id = pl.customer_id
        """)
        joined_customers = cursor.fetchone()[0]
        print(f"   Unique customers after JOIN: {joined_customers:,}")
        
        # 6. Check for missing Turkish customers in JOIN
        cursor.execute("""
            SELECT cd.nationality, COUNT(*) as count
            FROM casino_data.customer_demographics cd
            INNER JOIN casino_data.promo_label pl ON cd.customer_id = pl.customer_id
            WHERE cd.nationality IS NOT NULL
            GROUP BY cd.nationality 
            ORDER BY count DESC
            LIMIT 15
        """)
        
        joined_nationalities = cursor.fetchall()
        print(f"\n   Top 15 nationalities after JOIN:")
        for nationality, count in joined_nationalities:
            print(f"     {nationality}: {count:,}")
        
        # 7. Specific Turkey check in JOIN
        cursor.execute("""
            SELECT cd.nationality, COUNT(*) as count
            FROM casino_data.customer_demographics cd
            INNER JOIN casino_data.promo_label pl ON cd.customer_id = pl.customer_id
            WHERE cd.nationality ILIKE '%turkey%' OR cd.nationality ILIKE '%turkish%'
            GROUP BY cd.nationality
        """)
        
        turkey_joined = cursor.fetchall()
        print(f"\n   Turkey/Turkish in JOIN results:")
        if turkey_joined:
            for nationality, count in turkey_joined:
                print(f"     {nationality}: {count:,}")
        else:
            print(f"     No Turkey/Turkish found in JOIN")
        
        # 8. Check for NULL nationalities
        cursor.execute("""
            SELECT COUNT(*) 
            FROM casino_data.customer_demographics 
            WHERE nationality IS NULL
        """)
        null_nationalities = cursor.fetchone()[0]
        print(f"\n   NULL nationalities: {null_nationalities:,}")
        
        # 9. Sample Turkish customer check
        cursor.execute("""
            SELECT customer_id, nationality, region 
            FROM casino_data.customer_demographics 
            WHERE nationality ILIKE '%turkey%' OR nationality ILIKE '%turkish%'
            LIMIT 5
        """)
        
        turkish_samples = cursor.fetchall()
        if turkish_samples:
            print(f"\n   Sample Turkish customers:")
            for customer_id, nationality, region in turkish_samples:
                print(f"     ID: {customer_id}, Nationality: {nationality}, Region: {region}")
                
                # Check if they have promotion data
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM casino_data.promo_label 
                    WHERE customer_id = %s
                """, (customer_id,))
                promo_count = cursor.fetchone()[0]
                print(f"       → Promotion records: {promo_count}")
        
        # 10. Alternative spellings check
        print(f"\n4. Alternative spellings check:")
        alternative_spellings = ['Turkey', 'turkey', 'Turkish', 'turkish', 'Türkiye', 'Turkiye']
        
        for spelling in alternative_spellings:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM casino_data.customer_demographics 
                WHERE nationality = %s
            """, (spelling,))
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"   '{spelling}': {count:,} customers")
        
        # 11. Check exact nationality values
        cursor.execute("""
            SELECT DISTINCT nationality 
            FROM casino_data.customer_demographics 
            WHERE nationality IS NOT NULL
            ORDER BY nationality
        """)
        
        all_nationalities = [row[0] for row in cursor.fetchall()]
        print(f"\n5. All nationality values in database:")
        print(f"   Total unique nationalities: {len(all_nationalities)}")
        
        # Look for Turkey-like entries
        turkey_like = [nat for nat in all_nationalities if 'turk' in nat.lower() or 'türk' in nat.lower()]
        if turkey_like:
            print(f"   Turkey-related entries found: {turkey_like}")
        else:
            print(f"   No Turkey-related entries found")
        
        # Show first 20 nationalities alphabetically
        print(f"\n   First 20 nationalities (alphabetical):")
        for i, nationality in enumerate(all_nationalities[:20]):
            print(f"     {i+1:2d}. {nationality}")
        
        conn.close()
        
        return {
            'total_demo': total_demo,
            'total_promo': total_promo,
            'joined_records': joined_records,
            'turkey_count': turkey_count,
            'all_nationalities': all_nationalities
        }
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("NATIONALITY DATA VALIDATION")
    print("Checking for Turkish customers and data integrity")
    print("=" * 50)
    
    result = validate_nationality_data()
    
    if result:
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Demographics table: {result['total_demo']:,} customers")
        print(f"   Promo label table: {result['total_promo']:,} records")
        print(f"   JOIN result: {result['joined_records']:,} records")
        print(f"   Turkish customers: {result['turkey_count']:,}")
        print(f"   Total nationalities: {len(result['all_nationalities'])}")
        
        if result['turkey_count'] == 0:
            print(f"\n⚠️  WARNING: No Turkish customers found!")
            print(f"   This may indicate:")
            print(f"   • Different spelling (Türkiye, Turkiye)")
            print(f"   • Data encoding issues")
            print(f"   • Missing demographic data for Turkish customers")
        else:
            print(f"\n✅ Turkish customers found in database")