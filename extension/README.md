# Prohori Chrome extension

Check Bangla text for AI-likeness without leaving the page. Select text, right-click, **Check selection with Prohori**.

## Install (unpacked, about 2 minutes, no developer account)

1. Open `chrome://extensions` in Chrome (or Edge, Brave: `edge://extensions`).
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose this `extension` folder.
4. Pin the Prohori icon from the puzzle-piece menu if you like.

## Use

| Action | How |
|---|---|
| Check selected text | Select text, right-click, **Check selection with Prohori**, or press **Alt+Shift+P** |
| Check a whole article | Right-click anywhere, **Check this page's main text with Prohori** |
| Paste text manually | Click the toolbar icon, paste, **Analyze** |
| Change the shortcut | `chrome://extensions/shortcuts` |
| Use your own server | Toolbar icon, Settings, enter your server address, Save |

A card appears at the top right of the page with the probability, the evidence level, the three measured reasons and a passage-by-passage view for long text. Press **Esc** to close it.

The first check after a quiet period can take up to a minute because the free demo server sleeps when idle; the card says so while it waits.

## Privacy

* The extension reads nothing and sends nothing until you trigger a check. It does **not** run on every page: it uses `activeTab` and injects its card only when you ask.
* The text you check is sent to the server address in Settings (default: the public demo) to be scored. It is not stored by the extension; there are no analytics.
* The result card lives in a closed shadow root, so the page cannot read or restyle it, and nothing from the server is ever parsed as HTML.
* Run your own server for sensitive text (see the main README). Full statement: [PRIVACY.md](PRIVACY.md).

## Limits

* It cannot read canvas-based editors (Google Docs, Figma), the Chrome Web Store, `chrome://` pages or the built-in PDF viewer. Paste the text into the popup instead.
* Text over 20,000 characters is clipped (the card says so).
* The score is a probability with known error rates (see the app's "How accurate is this?" panel). Never present a flag as proof.

## What is tested

`node extension/test.js` runs 21 offline checks (jsdom plus a mocked Chrome API): result rendering, hostile text shown literally and never parsed as HTML, the server wake-up retry loop, error relaying, text clipping, the context-menu and popup message paths. **It has not been run in a real Chrome by its author**; load it unpacked and try it once before sharing it.

## Publishing to the Chrome Web Store (optional)

1. Create a developer account (one-time registration fee, currently about 5 USD) at the Chrome Web Store developer dashboard.
2. Zip the **contents** of this folder (not the folder itself, and without `test.js`).
3. Upload, then fill in: description, at least one 1280x800 screenshot, and a **privacy policy URL** (host `PRIVACY.md` publicly, for example through GitHub Pages or the raw file URL).
4. Under "Privacy practices", declare that it handles *website content* the user selects, sends it to your server for scoring, and does not sell or share it. Justify `activeTab` (read the selection on demand), `scripting` (show the result card), `contextMenus`, `storage` (your server address) and the host permission for the server.
5. Review usually takes from a few days to a couple of weeks. The extension must keep working with whatever server address you ship as the default, so decide whether to point it at a paid, always-on host first.
