"""
Casino Customer Segmentation Research Configuration
===================================================
Author: Muhammed Yavuzhan CANLI
Institution: University of Bath, Department of Computer Science
Supervisor: Dr. Moody Alam
Ethics Approval: 10351-12382
Date: June 2025

This module contains configuration for academic compliance and 
ensures all research activities align with university standards.
"""

ACADEMIC_METADATA = {
    "institution": "University of Bath",
    "department": "Computer Science - Software Engineering",
    "student_id": "mycc21",
    "supervisor": "Dr. Moody Alam",
    "ethics_ref": "10351-12382",
    "data_classification": "ANONYMIZED_BUSINESS_DATA",
    "gdpr_compliance": "Article 4(5) Full Anonymization",
    "academic_year": "2024-2025",
    "submission_date": "September 2025"
}

# Header template shall be used in each file
def get_academic_header(module_name: str = ""):
    """Generate academic header for each module"""
    return f"""
{module_name}
{'=' * len(module_name)}
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: {ACADEMIC_METADATA['ethics_ref']}
Academic use only - No commercial distribution
"""