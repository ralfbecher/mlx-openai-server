"""
XML-based tool call parser for handling tool calls in XML format.

This parser handles two formats:

Format 1 (with wrapper):
<tool_call>
<function=function_name>
<parameter=param_name>value</parameter>
</function>
</tool_call>

Format 2 (without wrapper):
<function=function_name>
<parameter=param_name>value</parameter>
</function>

The parser auto-detects which format is being used.
"""
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET
from loguru import logger
from app.handler.parser.base import BaseToolParser, ParseToolState


class XMLToolParser(BaseToolParser):
    """Parser for XML-based tool response format."""

    def __init__(self):
        # Use <tool_call> as the primary opening marker
        super().__init__(
            tool_open="<tool_call>",
            tool_close="</tool_call>"
        )
        # Additional state for XML parsing
        self.xml_buffer = ""
        self.function_name = None  # Store function name during parsing
        self.no_wrapper_mode = False  # Track if we're parsing without <tool_call> wrapper
        self.partial_tag_buffer = ""  # Buffer for detecting partial tag matches

    def _parse_tool_content(self, tool_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse XML-formatted tool content.

        Expected format:
        <function=function_name>
        <parameter=param1>value1</parameter>
        <parameter=param2>value2</parameter>
        </function>

        Args:
            tool_content: The string content containing the function call.

        Returns:
            A dictionary with 'name' and 'arguments' keys, or None if parsing fails.
        """
        try:
            # Extract function name from <function=...> tag
            function_match = re.search(r'<function=([^>]+)>', tool_content)
            if not function_match:
                raise ValueError("No function tag found in tool content")

            function_name = function_match.group(1).strip()

            # Extract all parameters
            arguments = {}

            # Find all <parameter=...>...</parameter> pairs
            param_pattern = r'<parameter=([^>]+)>\s*([^<]*?)\s*</parameter>'
            param_matches = re.finditer(param_pattern, tool_content, re.DOTALL)

            for match in param_matches:
                param_name = match.group(1).strip()
                param_value = match.group(2).strip()

                # Try to parse as JSON if it looks like structured data
                # Otherwise keep as string
                try:
                    # Check if value looks like JSON
                    if param_value.startswith(('[', '{')):
                        arguments[param_name] = json.loads(param_value)
                    else:
                        arguments[param_name] = param_value
                except (json.JSONDecodeError, ValueError):
                    # Keep as string if JSON parsing fails
                    arguments[param_name] = param_value

            return {
                "name": function_name,
                "arguments": arguments
            }

        except Exception as e:
            logger.error(f"Error parsing XML tool content: {e}")
            logger.debug(f"Content was: {tool_content}")
            raise

    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for XML tool calls.

        This overrides the base implementation to handle XML-specific streaming.
        Supports both <tool_call> wrapped and unwrapped <function=> formats.

        Returns:
            Tuple[parsed_content, is_complete]:
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if tool call is complete
        """
        if chunk is None:
            return None, True

        # Check if we're starting a tool call with <tool_call> wrapper
        if self.tool_open in chunk and self.state == ParseToolState.NORMAL:
            logger.debug(f"[XMLToolParser] Found tool_open (<tool_call>) in chunk")
            self.state = ParseToolState.FOUND_PREFIX
            self.no_wrapper_mode = False
            start_tool_index = chunk.find(self.tool_open)

            # Check if the entire tool call is in this chunk
            end_tool_index = chunk.find(self.tool_close, start_tool_index + len(self.tool_open))
            if end_tool_index != -1:
                # Complete tool call in one chunk
                logger.debug(f"[XMLToolParser] Complete tool call in one chunk")
                # Extract content between <tool_call> and </tool_call>
                tool_content = chunk[start_tool_index + len(self.tool_open):end_tool_index]
                self.state = ParseToolState.NORMAL
                self.buffer = ""

                try:
                    # Parse the tool call content
                    result = self._parse_tool_content(tool_content)
                    logger.debug(f"[XMLToolParser] Returning complete tool call: {result['name']}")
                    self.function_name = None

                    # When a complete tool call is found in one chunk, return it directly
                    # Any text before the tool call is considered preamble/thinking and can be discarded
                    # since the tool call is the primary content
                    return result, True
                except Exception as e:
                    logger.error(f"[XMLToolParser] Error parsing tool call: {e}")
                    self.function_name = None
                    return None, True

            # Only opening tag found, start buffering content after <tool_call>
            self.buffer = chunk[start_tool_index + len(self.tool_open):]
            logger.debug(f"[XMLToolParser] Starting to buffer, buffer size: {len(self.buffer)}")

            # Return any content before the tool call
            before_tool = chunk[:start_tool_index]
            return before_tool if before_tool else None, False

        # Check if we're starting a tool call WITHOUT <tool_call> wrapper (direct <function=>)
        if "<function=" in chunk and self.state == ParseToolState.NORMAL:
            logger.debug(f"[XMLToolParser] Found <function=> (no wrapper mode)")
            self.state = ParseToolState.FOUND_PREFIX
            self.no_wrapper_mode = True
            start_func_index = chunk.find("<function=")

            # Check if the entire function call is in this chunk
            end_func_index = chunk.find("</function>", start_func_index)
            if end_func_index != -1:
                # Complete function call in one chunk
                logger.debug(f"[XMLToolParser] Complete function call in one chunk (no wrapper)")
                # Extract content from <function=> to </function>
                func_content = chunk[start_func_index:end_func_index + len("</function>")]
                self.state = ParseToolState.NORMAL
                self.buffer = ""

                try:
                    # Parse the function call content
                    result = self._parse_tool_content(func_content)
                    logger.debug(f"[XMLToolParser] Returning complete function call: {result['name']}")
                    self.function_name = None
                    self.no_wrapper_mode = False

                    # When a complete function call is found in one chunk, return it directly
                    # Any text before the function call is considered preamble/thinking and can be discarded
                    return result, True
                except Exception as e:
                    logger.error(f"[XMLToolParser] Error parsing function call: {e}")
                    self.function_name = None
                    self.no_wrapper_mode = False
                    return None, True

            # Only opening function tag found, start buffering
            self.buffer = chunk[start_func_index:]
            logger.debug(f"[XMLToolParser] Starting to buffer (no wrapper), buffer size: {len(self.buffer)}")

            # Return any content before the function call
            before_func = chunk[:start_func_index]
            return before_func if before_func else None, False

        # Currently buffering a tool call
        if self.state == ParseToolState.FOUND_PREFIX:
            if self.no_wrapper_mode:
                # Looking for </function> when in no-wrapper mode
                # Check in the combined buffer + chunk for token-by-token streaming
                combined = self.buffer + chunk
                end_func_index = combined.find("</function>")
                if end_func_index != -1:
                    # Found the closing tag
                    logger.debug(f"[XMLToolParser] Found </function>, completing function call (no wrapper)")
                    # Extract everything up to and including </function>
                    complete_function = combined[:end_func_index + len("</function>")]
                    self.state = ParseToolState.NORMAL
                    self.buffer = ""

                    try:
                        # Parse the function call content
                        result = self._parse_tool_content(complete_function)
                        logger.debug(f"[XMLToolParser] Returning complete function call: {result['name']}")
                        self.function_name = None
                        self.no_wrapper_mode = False
                        return result, True
                    except Exception as e:
                        logger.error(f"[XMLToolParser] Error parsing function call: {e}")
                        self.function_name = None
                        self.no_wrapper_mode = False
                        return None, False
                else:
                    # Still buffering
                    self.buffer = combined
                    logger.debug(f"[XMLToolParser] Still buffering (no wrapper), buffer size: {len(self.buffer)}")
                    return None, False
            else:
                # Looking for </tool_call> when in wrapper mode
                # Check in the combined buffer + chunk for token-by-token streaming
                combined = self.buffer + chunk
                end_tool_index = combined.find(self.tool_close)
                if end_tool_index != -1:
                    # Found the closing tag
                    logger.debug(f"[XMLToolParser] Found tool_close, completing tool call")
                    # Extract everything up to but not including </tool_call>
                    tool_content = combined[:end_tool_index]
                    self.state = ParseToolState.NORMAL
                    self.buffer = ""

                    try:
                        # Parse the tool call content
                        result = self._parse_tool_content(tool_content)
                        logger.debug(f"[XMLToolParser] Returning complete tool call: {result['name']}")
                        self.function_name = None
                        return result, True
                    except Exception as e:
                        logger.error(f"[XMLToolParser] Error parsing tool call: {e}")
                        self.function_name = None
                        return None, False
                else:
                    # Still buffering
                    self.buffer = combined
                    logger.debug(f"[XMLToolParser] Still buffering, buffer size: {len(self.buffer)}")
                    return None, False

        # Normal content, not in a tool call
        # Handle partial tag detection for token-by-token streaming
        # Combine partial buffer with new chunk
        combined = self.partial_tag_buffer + chunk

        # Check if combined text now contains opening tags
        if "<tool_call>" in combined:
            # Found opening tag after buffering
            self.partial_tag_buffer = ""
            return self.parse_stream(combined)
        elif "<function=" in combined:
            # Found opening tag after buffering
            self.partial_tag_buffer = ""
            return self.parse_stream(combined)

        # Check if the end of combined could be a partial tag
        # Tags we're looking for: "<tool_call>" and "<function="
        potential_tags = ["<tool_call>", "<function="]

        for tag in potential_tags:
            # Check if any prefix of the tag matches the end of combined
            # Check from longest to shortest prefix
            for i in range(len(tag) - 1, 0, -1):
                if combined.endswith(tag[:i]):
                    # This could be the start of a tag, buffer it
                    logger.debug(f"[XMLToolParser] Buffering potential tag start: {repr(tag[:i])}")
                    self.partial_tag_buffer = tag[:i]
                    # Return everything except the potential tag start
                    content_to_return = combined[:-i]
                    return content_to_return if content_to_return else None, False

        # No partial match found, return all as normal content
        if self.partial_tag_buffer:
            self.partial_tag_buffer = ""

        logger.debug(f"[XMLToolParser] Normal content: {repr(combined[:50])}")
        return combined, False
