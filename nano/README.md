# 🍌 Nano Banana — AI Image Transformer

A simple full-stack web app that performs AI image-to-image transformation using
Google's Gemini API (the "Nano Banana" model, `gemini-2.0-flash-exp`).

Upload an image, describe the change you want, and get a transformed image back —
ready to preview and download.

## Features

- Drag-and-drop image upload (max 20MB)
- Free-text transformation prompt
- One-click generation with a loading spinner
- Preview + download of the result
- Clean, minimal, mobile-responsive dark UI
- API key stored locally in `.env` (entered once on first launch)

## Tech stack

- **Backend:** Python Flask
- **AI:** Google `google-genai` SDK (`gemini-2.0-flash-exp`)
- **Frontend:** Vanilla HTML / CSS / JS — no frameworks

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your API key** (either way works)

   - Easiest: just run the app — on first load it asks for your key and saves it
     to `.env` automatically.
   - Or manually copy the example and edit it:

     ```bash
     cp .env.example .env
     # then set GOOGLE_API_KEY=your_key_here
     ```

   Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

3. **Run**

   ```bash
   python app.py
   ```

   Open <http://localhost:5000> in your browser.

## How it works

1. On first load you enter your Gemini API key; it's written to `.env`.
2. Drag an image into the left panel.
3. Type what you want changed in the prompt box.
4. Click **Generate** — the image + prompt are sent to Gemini.
5. The result appears on the right; click **Download** to save it.

Generated images are saved to the `outputs/` folder with unique filenames.

## Notes

- One image at a time. No login, no database, no model selector.
- If the API fails or no image is returned, a friendly error message is shown.
