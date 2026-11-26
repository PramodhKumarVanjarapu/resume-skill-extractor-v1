"""Project splitting on layouts seen in real resumes (synthetic text, no personal data)."""

import unittest

from project_extractor import ProjectExtractor
from technology_matcher import TechnologyMatcher


class ProjectLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = ProjectExtractor(TechnologyMatcher())

    def names(self, text):
        return [p.name for p in self.ext.extract(text).projects]

    def test_right_aligned_technology_column(self):
        text = ("Hotel Booking System                          Java 21, Spring Boot, MySQL\n"
                "  • Built room search and booking APIs.\n"
                "Online Quiz Platform                           Spring Boot, React.js, PostgreSQL\n"
                "  • Built timed quizzes with a leaderboard.\n")
        result = self.ext.extract(text)
        self.assertEqual([p.name for p in result.projects], ["Hotel Booking System", "Online Quiz Platform"])
        self.assertIn("MySQL", result.projects[0].technologies)
        self.assertNotIn("MySQL", result.projects[1].technologies)
        self.assertIn("PostgreSQL", result.projects[1].technologies)

    def test_pipe_titles_indented_less_than_bullets(self):
        # The section's first title loses its indent when the section text is trimmed.
        text = ("Career Portal | React.js, Node.js, MongoDB                 2026\n"
                "       • Developed a job portal for seekers and recruiters.\n"
                "  Shopping-App | React.js, CSS                              2026\n"
                "       • Developed a shopping cart.\n"
                "  E-commerce App | React.js, Express.js                     2025\n"
                "       • Implemented catalog and orders.\n")
        self.assertEqual(self.names(text), ["Career Portal", "Shopping-App", "E-commerce App"])

    def test_symbol_font_bullets_are_bullets(self):
        text = ("CarHub — React.js, Node.js, MySQL\n"
                " Built a car service booking platform.\n"
                " To login use this Username: demo, Password: demo.\n"
                "Jobby Portal — React.js, REST API\n"
                " Built job search with filters.\n")
        self.assertEqual(self.names(text), ["CarHub", "Jobby Portal"])

    def test_wrapped_title_and_wrapped_description(self):
        text = ("AI CRM (Python, FastAPI, LangGraph, React, Redux\n"
                "Toolkit)                                          GitHub\n"
                "  • Built a FastAPI backend that logs interactions and keeps separate task boards with login and\n"
                "separate boards for every team member.\n"
                "Luna Assistant (Python, Ollama, Electron)\n"
                "  • Built a local assistant.\n")
        self.assertEqual(self.names(text), ["AI CRM", "Luna Assistant"])

    def test_link_words_removed_from_names(self):
        text = ("MindVault [GitHub] [Live]\n"
                "  • Built a notes app with search.\n\n"
                "ERP System(GitHub: https://github.com/example/erp)\n"
                "  • Built inventory and billing modules.\n\n"
                "Chat Application with RAG GitHub          Python, FastAPI, ChromaDB\n"
                "  • Built a document chatbot.\n")
        self.assertEqual(self.names(text), ["MindVault", "ERP System", "Chat Application with RAG"])

    def test_link_only_and_highlight_lines_are_not_projects(self):
        text = ("Movie Production AI\n"
                "Live Demo\n"
                "  • Built a script analysis tool.\n"
                "Highlights\n"
                "  • Reduced latency by 40%.\n")
        self.assertEqual(self.names(text), ["Movie Production AI"])

    def test_one_line_projects_with_their_own_links(self):
        text = ("Additional Projects\n"
                "● Airline Booking System — Python, MySQL — GitHub\n"
                "● Super App — Personalized Dashboard — React, Node.js — GitHub | Live Demo\n")
        result = self.ext.extract(text)
        self.assertEqual([p.name for p in result.projects],
                         ["Airline Booking System", "Super App — Personalized Dashboard"])
        self.assertIn("MySQL", result.projects[0].technologies)
        self.assertIn("Node.js", result.projects[1].technologies)

    def test_technology_line_is_not_a_project(self):
        text = ("StayNexus [GitHub]\n"
                " Spring Boot | REST APIs | PostgreSQL | JWT\n"
                " – Architected a rental backend with 10+ REST APIs.\n"
                "Expense Tracker\n"
                " WebSocket                                                 2024–25\n"
                " – Built live expense sync.\n")
        self.assertEqual(self.names(text), ["StayNexus", "Expense Tracker"])

    def test_technologies_glued_to_the_name(self):
        text = ("AI-Powered Quiz Generator React.js, FastAPI, Python, PostgreSQL\n"
                "  • Generates quizzes from PDFs.\n\n"
                "Student Management System (Core Java, MySQL, JDBC)\n"
                "  • CRUD for student records.\n")
        self.assertEqual(self.names(text), ["AI-Powered Quiz Generator", "Student Management System"])


if __name__ == "__main__":
    unittest.main()
