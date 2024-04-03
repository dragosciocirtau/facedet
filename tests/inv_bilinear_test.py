import unittest
from face_detection.facedet import inv_bilinear

class TestInvBilinear(unittest.TestCase):
    def test_square(self):
        corners = [(1,4), (1, 1), (4, 1), (4, 4)]
        
        self.assertEqual(inv_bilinear(corners[0], corners), (0, 0))
        self.assertEqual(inv_bilinear(corners[1], corners), (0, 1))
        self.assertEqual(inv_bilinear(corners[2], corners), (1, 1))
        self.assertEqual(inv_bilinear(corners[3], corners), (1, 0))
    
    
    def test_paralelogram(self):
        corners = [(2,4), (1, 1), (4, 1), (5, 4)]
        
        self.assertEqual(inv_bilinear(corners[0], corners), (0, 0))
        self.assertEqual(inv_bilinear(corners[1], corners), (0, 1))
        self.assertEqual(inv_bilinear(corners[2], corners), (1, 1))
        self.assertEqual(inv_bilinear(corners[3], corners), (1, 0))
        
    
    def test_iregular_trapeze(self):
        corners = [(2,5), (1, 2), (4, 0), (6, 4)]
        
        self.assertEqual(inv_bilinear(corners[0], corners), (0, 0))
        self.assertEqual(inv_bilinear(corners[1], corners), (0, 1))
        self.assertEqual(inv_bilinear(corners[2], corners), (1, 1))
        self.assertEqual(inv_bilinear(corners[3], corners), (1, 0))
    

if __name__ == '__main__':
    unittest.main()