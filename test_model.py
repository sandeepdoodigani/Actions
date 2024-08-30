import unittest
from model import model, X_test, y_test

class TestModel(unittest.TestCase):
    def test_model_accuracy(self):
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        self.assertGreater(accuracy, 0.9, "Accuracy should be greater than 0.9")

if __name__ == '__main__':
    unittest.main()
