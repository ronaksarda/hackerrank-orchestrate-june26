import unittest
from pipeline.schema import normalize_object_part
from pipeline.analyzer import merge_risk_flags, generate_cache_key

class TestLogic(unittest.TestCase):

    def test_normalize_object_part(self):
        # Car tests
        self.assertEqual(normalize_object_part("front_bumper", "car"), "front_bumper")
        self.assertEqual(normalize_object_part("screen", "car"), "unknown")
        
        # Laptop tests
        self.assertEqual(normalize_object_part("screen", "laptop"), "screen")
        self.assertEqual(normalize_object_part("front_bumper", "laptop"), "unknown")
        
        # Package tests
        self.assertEqual(normalize_object_part("box", "package"), "box")
        self.assertEqual(normalize_object_part("door", "package"), "unknown")

    def test_cache_key_generation(self):
        key1 = generate_cache_key("u1", "img1.jpg", "car", "claim text", "hashA")
        key2 = generate_cache_key("u1", "img1.jpg", "car", "claim text", "hashA")
        key3 = generate_cache_key("u1", "img1.jpg", "car", "claim text", "hashB")
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_merge_risk_flags(self):
        # Basic merge
        self.assertEqual(
            merge_risk_flags("blurry_image", False, "none", "true"),
            "blurry_image"
        )
        
        # User risk triggers manual review
        self.assertEqual(
            merge_risk_flags("none", True, "none", "true"),
            "manual_review_required;user_history_risk"
        )
        
        # Serious flag triggers manual review
        self.assertEqual(
            merge_risk_flags("claim_mismatch", False, "none", "true"),
            "claim_mismatch;manual_review_required"
        )
        
        # Invalid image triggers manual review
        self.assertEqual(
            merge_risk_flags("none", False, "none", "false"),
            "manual_review_required"
        )

if __name__ == '__main__':
    unittest.main()
