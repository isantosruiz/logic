import unittest

from api.index import app


class SimplificationEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def simplify(self, equation: str, result_form: str = "sop") -> dict:
        response = self.client.post(
            "/api/simplify",
            json={
                "mode": "equation",
                "equation": equation,
                "result_form": result_form,
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_complement_or_is_true_in_both_forms(self):
        for result_form in ("sop", "pos"):
            with self.subTest(result_form=result_form):
                result = self.simplify("A + ~A", result_form)
                self.assertEqual(result["simplified_expression"], "True")
                self.assertEqual(result["variables"], ["A"])

    def test_complement_and_is_false_in_both_forms(self):
        for result_form in ("sop", "pos"):
            with self.subTest(result_form=result_form):
                result = self.simplify("A * ~A", result_form)
                self.assertEqual(result["simplified_expression"], "False")
                self.assertEqual(result["variables"], ["A"])


if __name__ == "__main__":
    unittest.main()
