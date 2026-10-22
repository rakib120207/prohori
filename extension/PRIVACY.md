# Prohori extension: privacy statement

**What it reads:** only the text you select, or the main text of the page you explicitly ask it to check, at the moment you ask (right-click menu, keyboard shortcut, or the popup's Analyze button). It does not read pages in the background.

**Where it goes:** that text is sent over HTTPS to the Prohori server whose address is shown in the extension's Settings (default: the public demo at prohori-mhuf.onrender.com), solely to compute the score. The server returns the result and the extension shows it.

**What is kept:** the extension stores only the server address you chose (in browser sync storage). It does not store, log or transmit the text you check anywhere else, and it contains no analytics, advertising or tracking.

**Server side:** the public demo server computes the score in memory and does not persist submitted text in the application. Its hosting provider may keep standard infrastructure logs. If you handle sensitive text, run your own server and enter its address in Settings.

**Contact:** open an issue at https://github.com/rakib120207/prohori
