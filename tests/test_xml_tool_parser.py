import unittest
from app.handler.parser.xml_tool import XMLToolParser


class TestXMLToolParser(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "name": "simple XML function call",
                "chunks": '''<tool_call>
<function=connect_database_mcp_Orionbelt-Semantic-Layer>
<parameter=db_type>
postgresql
</parameter>
</function>
</tool_call>'''.split('\n'),
                "expected_name": "connect_database_mcp_Orionbelt-Semantic-Layer",
                "expected_args": {"db_type": "postgresql"}
            },
            {
                "name": "XML function call with multiple parameters",
                "chunks": '''<tool_call>
<function=execute_query>
<parameter=db_type>
mysql
</parameter>
<parameter=query>
SELECT * FROM users
</parameter>
<parameter=limit>
10
</parameter>
</function>
</tool_call>'''.split('\n'),
                "expected_name": "execute_query",
                "expected_args": {
                    "db_type": "mysql",
                    "query": "SELECT * FROM users",
                    "limit": "10"
                }
            },
            {
                "name": "streaming XML function call",
                "chunks": [
                    '<tool_call>',
                    '<function=get_weather>',
                    '<parameter=city>',
                    'London',
                    '</parameter>',
                    '<parameter=units>',
                    'metric',
                    '</parameter>',
                    '</function>',
                    '</tool_call>'
                ],
                "expected_name": "get_weather",
                "expected_args": {
                    "city": "London",
                    "units": "metric"
                }
            }
        ]

    def test_parse_complete(self):
        """Test parsing complete XML tool calls"""
        parser = XMLToolParser()

        # Test case 1: Complete tool call in one string
        content = '''<function=connect_database_mcp_Orionbelt-Semantic-Layer>
<parameter=db_type>
postgresql
</parameter>
</function>'''

        result = parser._parse_tool_content(content)

        self.assertEqual(result["name"], "connect_database_mcp_Orionbelt-Semantic-Layer")
        self.assertIsInstance(result["arguments"], dict)
        self.assertEqual(result["arguments"]["db_type"], "postgresql")

    def test_parse_stream(self):
        """Test parsing streaming XML tool calls"""
        for test_case in self.test_cases:
            with self.subTest(msg=test_case["name"]):
                parser = XMLToolParser()
                final_result = None

                for chunk in test_case["chunks"]:
                    content, is_complete = parser.parse_stream(chunk)

                    if is_complete and content and isinstance(content, dict):
                        final_result = content
                        break

                self.assertIsNotNone(final_result, f"No result found for {test_case['name']}")
                self.assertEqual(final_result["name"], test_case["expected_name"])

                # Arguments should be a dict, not a JSON string
                self.assertIsInstance(final_result["arguments"], dict)
                self.assertEqual(final_result["arguments"], test_case["expected_args"])

    def test_single_chunk_complete_call(self):
        """Test parsing a complete tool call in a single chunk"""
        parser = XMLToolParser()

        chunk = '''<tool_call>
<function=test_function>
<parameter=param1>
value1
</parameter>
</function>
</tool_call>'''

        content, is_complete = parser.parse_stream(chunk)

        self.assertTrue(is_complete)
        self.assertIsNotNone(content)
        self.assertEqual(content["name"], "test_function")

        # Arguments should be a dict, not a JSON string
        self.assertIsInstance(content["arguments"], dict)
        self.assertEqual(content["arguments"]["param1"], "value1")

    def test_multi_chunk_call(self):
        """Test parsing a tool call split across multiple chunks"""
        parser = XMLToolParser()

        chunks = [
            '<tool_call>',
            '<function=my_func>',
            '<parameter=key>',
            'my_value',
            '</parameter>',
            '</function>',
            '</tool_call>'
        ]

        result = None
        for chunk in chunks:
            content, is_complete = parser.parse_stream(chunk)
            if is_complete and content:
                result = content
                break

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "my_func")

        # Arguments should be a dict, not a JSON string
        self.assertIsInstance(result["arguments"], dict)
        self.assertEqual(result["arguments"]["key"], "my_value")

    def test_without_tool_call_wrapper(self):
        """Test parsing XML format WITHOUT <tool_call> wrapper (actual model output)"""
        parser = XMLToolParser()

        # Test case: Model outputs function call directly without <tool_call> wrapper
        chunk = '''<function=connect_database_mcp_Orionbelt-Semantic-Layer>
<parameter=db_type>
postgresql
</parameter>
</function>'''

        content, is_complete = parser.parse_stream(chunk)

        self.assertTrue(is_complete)
        self.assertIsNotNone(content)
        self.assertEqual(content["name"], "connect_database_mcp_Orionbelt-Semantic-Layer")

        # Arguments should be a dict, not a JSON string
        self.assertIsInstance(content["arguments"], dict)
        self.assertEqual(content["arguments"]["db_type"], "postgresql")

    def test_streaming_without_wrapper(self):
        """Test streaming XML format WITHOUT <tool_call> wrapper"""
        parser = XMLToolParser()

        chunks = [
            '<function=get_weather>',
            '<parameter=city>',
            'London',
            '</parameter>',
            '<parameter=units>',
            'metric',
            '</parameter>',
            '</function>'
        ]

        result = None
        for chunk in chunks:
            content, is_complete = parser.parse_stream(chunk)
            if is_complete and content and isinstance(content, dict):
                result = content
                break

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "get_weather")

        # Arguments should be a dict, not a JSON string
        self.assertIsInstance(result["arguments"], dict)
        self.assertEqual(result["arguments"]["city"], "London")
        self.assertEqual(result["arguments"]["units"], "metric")

    def test_text_before_function_call(self):
        """Test that text before a function call is handled (tool call is returned, preamble discarded)"""
        parser = XMLToolParser()

        chunk = '''I'll help you explore the database.

<function=connect_database>
<parameter=db_type>postgresql</parameter>
</function>'''

        content, is_complete = parser.parse_stream(chunk)

        # When function call is complete in one chunk, it returns the tool call directly
        # Text before the tool call is considered preamble/thinking and is discarded
        # since the tool call is the primary actionable content
        self.assertTrue(is_complete)
        self.assertIsNotNone(content)
        self.assertIsInstance(content, dict)
        self.assertEqual(content["name"], "connect_database")

        # Arguments should be a dict, not a JSON string
        self.assertIsInstance(content["arguments"], dict)
        self.assertEqual(content["arguments"]["db_type"], "postgresql")


if __name__ == '__main__':
    unittest.main()
