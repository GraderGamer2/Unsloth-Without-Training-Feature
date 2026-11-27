# Unsloth Without Training Feature

**Independent, unofficial inference-only fork maintained by [GraderGamer2](https://github.com/GraderGamer2).** This is not an official Unsloth AI release, product or support channel, and is not endorsed by or affiliated with Unsloth AI.

## Personal Linux build

The maintained build branch is [`ryzen-7735hs-vulkan`](https://github.com/GraderGamer2/Unsloth-Without-Training-Feature/tree/ryzen-7735hs-vulkan). It targets the AMD Ryzen 7 7735HS and Radeon 680M on Debian 13 x86-64 with Wayland and Vulkan. The AppImage is intended for local inference, not model training.

- Local GGUF inference, model downloads and external API providers.
- Chat, agents, tool calling, MCP, web research and retrieval-augmented generation.
- Image and audio inference and audio-file transcription.
- Machine-specific compiler optimization, reduced diagnostic output, and independently cached native builds.

Download this fork's AppImage from [GitHub Actions](https://github.com/GraderGamer2/Unsloth-Without-Training-Feature/actions/workflows/ryzen-vulkan-appimage.yml). Open a successful run on the maintained branch and download the `Unsloth-Ryzen-7735HS-Vulkan` artifact. Verify the included SHA256 checksum before running it. No upstream installer or release is required. Existing personal-build data remains under `~/.unsloth`.

Report fork-specific problems through [this repository's issues](https://github.com/GraderGamer2/Unsloth-Without-Training-Feature/issues).

## Origin and licensing

This project is derived from [Unsloth](https://github.com/unslothai/unsloth), developed by its original authors and contributors, and incorporates upstream native inference engines. The fork modifications are maintained by GraderGamer2. Original copyright notices, license texts and third-party attribution remain applicable. See [LICENSE](LICENSE), [COPYING](COPYING), and [Studio's AGPL license](studio/LICENSE.AGPL-3.0).
