#!/usr/bin/env python3
"""Lane GW (2026-10-02): the gateway knows the default local stack is text-only.  An image/video/audio part is routed to
a declared vision provider (inside the hard daily cap) or refused with one clear 400 -- never the engine's bare
'At most 0 image(s)'.  /v1/models advertises the modalities.  Run: python -m unittest test_gateway_text_only"""
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_gateway_local_first as lf

shim = lf.shim
IMG = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]


class Detect(unittest.TestCase):
    def test_media_parts(self):
        self.assertEqual(shim.request_media(json.dumps({"messages": IMG}).encode()), {"image"})
        self.assertEqual(shim.request_media(json.dumps({"messages": [
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}, {"type": "video_url"}]}]}).encode()),
            {"audio", "video"})
        self.assertEqual(shim.request_media(json.dumps({"input": [{"role": "user", "content": [{"type": "input_image"}]}]}).encode()), {"image"})
        self.assertEqual(shim.request_media(json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()), set())
        self.assertEqual(shim.request_media(b"not json"), set())
        self.assertEqual(shim.request_media(json.dumps({"messages": [{"role": "user", "content": "image_url"}]}).encode()), set())


class Guard(lf.RoutingBase):
    def setUp(self):
        super().setUp()
        for name, value in dict(LOCAL_MODALITIES="text", REMOTE_VISION=0).items():
            self.stack.enter_context(patch.object(shim, name, value))
        self.ptok = 3_000

    async def test_no_remote_vision_gives_one_clear_400_and_never_touches_the_engine(self):
        r = await self.route(messages=IMG)
        self.assertEqual(r.status, 400)
        err = json.loads(r.body)["error"]
        self.assertEqual(err["code"], "local_model_text_only")
        self.assertIn("text-only", err["message"])
        self.assertEqual(self.calls, [])

    async def test_vision_remote_is_used_when_declared_and_the_cap_has_room(self):
        with patch.object(shim, "REMOTE_VISION", 1):
            self.assertEqual(await self.route(messages=IMG), "remote")
        self.assertEqual((self.calls, self.events), (["remote"], [("remote", "vision")]))

    async def test_vision_remote_respects_the_hard_cap(self):
        with patch.object(shim, "REMOTE_VISION", 1), patch.object(shim, "_spend_allows_overflow", lambda p, m: False):
            r = await self.route(messages=IMG)
        self.assertEqual(r.status, 503)
        self.assertEqual(self.calls, [])

    async def test_pinned_local_callers_never_go_remote_for_vision(self):
        with patch.object(shim, "REMOTE_VISION", 1):
            r = await self.route(model="estate-local", messages=IMG)
            self.assertEqual(r.status, 400)
            r = await self.route(headers={"X-Client": "coder-local-pin"}, messages=IMG)
            self.assertEqual(r.status, 400)
        self.assertEqual(self.calls, [])

    async def test_video_and_audio_are_not_sent_to_an_image_only_remote(self):
        with patch.object(shim, "REMOTE_VISION", 1):
            r = await self.route(messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {}}]}])
        self.assertEqual(r.status, 400)
        self.assertEqual(self.calls, [])

    async def test_vision_stack_declared_local_passes_images_through(self):
        with patch.object(shim, "LOCAL_MODALITIES", "image,text"):
            self.assertEqual(await self.route(messages=IMG), "local")
        self.assertEqual(self.calls, ["local"])

    async def test_text_requests_are_unaffected(self):
        self.assertEqual(await self.route(), "local")

    async def test_modalities_are_advertised_on_v1_models(self):
        rows = {r["id"]: r for r in shim._alias_model_rows()}
        self.assertEqual(rows["estate-local"]["input_modalities"], ["text"])
        self.assertEqual(rows["estate"]["input_modalities"], ["text"])
        self.assertFalse(rows["estate-local"]["capabilities"]["vision"])
        with patch.object(shim, "REMOTE_VISION", 1):
            rows = {r["id"]: r for r in shim._alias_model_rows()}
            self.assertEqual(rows["estate"]["input_modalities"], ["image", "text"])
            self.assertEqual(rows["estate-local"]["input_modalities"], ["text"])

        class Resp:
            body = json.dumps({"data": [{"id": "qwen-local", "max_model_len": 524288}]}).encode()
        with patch.object(shim, "_passthrough", AsyncMock(return_value=Resp())):
            out = json.loads((await shim.h_models(None)).body)
        local = [r for r in out["data"] if r["id"] == "qwen-local"][0]
        self.assertEqual(local["input_modalities"], ["text"])
        self.assertEqual(local["modalities"], {"input": ["text"], "output": ["text"]})
        self.assertEqual(local["max_model_len"], 524288)

    def test_config_roundtrip(self):
        with patch.object(shim, "_config_owner", lambda: False):
            shim.apply_config({"local_modalities": "text, Image", "remote_vision": "1"})
        self.assertEqual(shim.LOCAL_MODALITIES, "image,text")
        self.assertEqual(shim.REMOTE_VISION, 1)


if __name__ == "__main__":
    unittest.main()
