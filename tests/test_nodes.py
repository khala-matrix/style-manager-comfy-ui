import base64
import io
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import GPTImageGenerate


class TestBuildPrompt:
    def setup_method(self):
        self.node = GPTImageGenerate()

    def test_both_inputs(self):
        result = self.node.build_prompt("external design req", "style: minimalist")
        assert result == "external design req\nstyle: minimalist"

    def test_external_only(self):
        result = self.node.build_prompt("external design req", "")
        assert result == "external design req"

    def test_prompt_only(self):
        result = self.node.build_prompt("", "local prompt text")
        assert result == "local prompt text"

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="at least one text input"):
            self.node.build_prompt("", "")

    def test_none_external(self):
        result = self.node.build_prompt(None, "local prompt")
        assert result == "local prompt"

    def test_whitespace_only_treated_as_empty(self):
        with pytest.raises(ValueError, match="at least one text input"):
            self.node.build_prompt("   ", "   ")


class TestImageDecode:
    def setup_method(self):
        self.node = GPTImageGenerate()

    def _make_fake_response(self, width=64, height=64, n=1, mode="RGB"):
        images_data = []
        for i in range(n):
            img = Image.new(mode, (width, height), color=(255, 0, 0) if i == 0 else (0, 255, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            mock_item = MagicMock()
            mock_item.b64_json = b64
            images_data.append(mock_item)

        mock_response = MagicMock()
        mock_response.data = images_data
        return mock_response

    @patch("nodes.OpenAI")
    def test_single_image_output_shape(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.images.generate.return_value = self._make_fake_response(64, 64, n=1)

        result = self.node.generate(
            api_key="test-key", base_url="https://api.openai.com/v1",
            prompt="test", size="1024x1024",
            quality="auto", background="opaque", n=1, output_format="png"
        )

        assert result[0].shape == (1, 64, 64, 3)
        assert result[0].dtype == torch.float32
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    @patch("nodes.OpenAI")
    def test_batch_output_shape(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.images.generate.return_value = self._make_fake_response(64, 64, n=3)

        result = self.node.generate(
            api_key="test-key", base_url="https://api.openai.com/v1",
            prompt="test", size="1024x1024",
            quality="auto", background="opaque", n=3, output_format="png"
        )

        assert result[0].shape == (3, 64, 64, 3)

    @patch("nodes.OpenAI")
    def test_transparent_background_rgba(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.images.generate.return_value = self._make_fake_response(64, 64, n=1, mode="RGBA")

        result = self.node.generate(
            api_key="test-key", base_url="https://api.openai.com/v1",
            prompt="test", size="1024x1024",
            quality="auto", background="transparent", n=1, output_format="png"
        )

        assert result[0].shape == (1, 64, 64, 4)
