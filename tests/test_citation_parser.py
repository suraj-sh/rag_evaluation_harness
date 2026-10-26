from unittest import TestCase, main
from evaluation.score_answers import extract_cited_pages


class ExtractCitedPagesTests(TestCase):
    def test_single_page(self):
        self.assertEqual(extract_cited_pages("Claim [Page 24]."), {24})

    def test_multiple_pages(self):
        self.assertEqual(
            extract_cited_pages("Claim [Pages 24, 25]."),
            {24, 25},
        )

    def test_page_range(self):
        self.assertEqual(
            extract_cited_pages("Claim [Pages 37-38]."),
            {37, 38},
        )

    def test_range_and_single_page(self):
        self.assertEqual(
            extract_cited_pages("Claim [Pages 24, 37-39]."),
            {24, 37, 38, 39},
        )

    def test_duplicate_citations(self):
        self.assertEqual(
            extract_cited_pages(
                "First claim [Page 24]. Second claim [Page 24]."
            ),
            {24},
        )

    def test_no_citations(self):
        self.assertEqual(
            extract_cited_pages("There are no page citations here."),
            set(),
        )

    def test_reverse_range_is_ignored(self):
        self.assertEqual(extract_cited_pages("Claim [Pages 38-37]."), set())


if __name__ == "__main__":
    main()