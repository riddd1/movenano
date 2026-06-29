const MAX_BYTES = 20 * 1024 * 1024;
const COST_PER_IMAGE = 0.039;
const MAX_CONCURRENT = 2; // how many generations run at the same time (gentler on the connection)

// Reusable preset prompts (dropdown above each prompt box).
const PRESETS = {
  bg: `Remove any text, captions, play buttons, social media icons, phone status bar elements (battery icon, signal bars, wifi icon, clock/time, carrier name, notch, dynamic island), and any other UI overlay from the image. Fill those areas with what would naturally be behind them.

Recreate this exact photo as a direct reference. Keep ALL clothing items the exact same color, style, and type. Keep all brand logos on clothing and shoes. Keep the same camera angle and overall composition.

Change ONLY:
- The background surface — if it is a bedsheet change to a different style and color bedsheet, if it is a floor change to a different floor type and color, if it is a table change to a different table surface
- Slightly reposition any small items like scrunchies, beauty products, sunglasses, or accessories so the layout feels fresh but still natural
- Replace any accessories (jewelry, watches, hair clips, beauty products) with different alternatives that match and complement the outfit's color palette

Soft natural window daylight, muted true-to-life colors, gentle realistic contact shadows under fabric edges. Visible fabric texture, natural wrinkles, fold lines and seams. Casual handheld iPhone snapshot, slightly soft not tack-sharp, faint sensor grain, warm indoor auto white balance. Avoid: oversaturated colors, HDR glow, harsh studio lighting, 3D render, CGI, plastic look. The output must contain zero text and zero icons.`,

  color: `Remove any text, captions, play buttons, social media icons, phone status bar elements (battery icon, signal bars, wifi icon, clock/time, carrier name, notch, dynamic island), and any other UI overlay from the image. Fill those areas with what would naturally be behind them.

Recreate this exact photo as a direct reference. Keep the EXACT same background, surface, camera angle, composition, and layout. Do NOT change the background at all. Keep all brand logos on clothing and shoes.

Change ONLY:
- Change the color of the clothing items to a different attractive color that is NOT the current color — pick a color that coordinates well across all visible garments so the outfit still looks intentional and stylish together
- Replace any accessories (jewelry, watches, hair clips, beauty products) with different alternatives that match and complement the NEW clothing color
- Keep all clothing styles, types, fits, folds, and positions identical — only the color changes

Soft natural window daylight, muted true-to-life colors, gentle realistic contact shadows under fabric edges. Visible fabric texture, natural wrinkles, fold lines and seams. Casual handheld iPhone snapshot, slightly soft not tack-sharp, faint sensor grain, warm indoor auto white balance. Avoid: oversaturated colors, HDR glow, harsh studio lighting, 3D render, CGI, plastic look. The output must contain zero text and zero icons.`,

  remix: `Remove any text, captions, play buttons, social media icons, phone status bar elements (battery icon, signal bars, wifi icon, clock/time, carrier name, notch, dynamic island), and any other UI overlay from the image. Fill those areas with what would naturally be behind them.

Recreate this exact photo as a direct reference. Keep the same camera angle and overall composition. Keep all brand logos on clothing and shoes.

Change ALL of these:
- The background surface — if it is a bedsheet change to a different style and color bedsheet, if it is a floor change to a different floor type and color, if it is a table change to a different table surface
- Change the color of the clothing items to a different attractive color that is NOT the current color — pick a color that coordinates well across all visible garments so the outfit still looks intentional and stylish together
- Slightly reposition any small items like scrunchies, beauty products, sunglasses, or accessories so the layout feels fresh but still natural
- Replace any accessories (jewelry, watches, hair clips, beauty products) with different alternatives that match and complement the NEW clothing color and NEW background
- Keep all clothing styles, types, fits, folds, and positions identical — only the colors change

Soft natural window daylight, muted true-to-life colors, gentle realistic contact shadows under fabric edges. Visible fabric texture, natural wrinkles, fold lines and seams. Casual handheld iPhone snapshot, slightly soft not tack-sharp, faint sensor grain, warm indoor auto white balance. Avoid: oversaturated colors, HDR glow, harsh studio lighting, 3D render, CGI, plastic look. The output must contain zero text and zero icons.`,
};

const setupEl = document.getElementById("setup");
const appEl = document.getElementById("app");
const apiKeyInput = document.getElementById("apiKeyInput");
const saveKeyBtn = document.getElementById("saveKeyBtn");
const setupError = document.getElementById("setupError");

const jobsEl = document.getElementById("jobs");
const jobTemplate = document.getElementById("jobTemplate");
const addJobBtn = document.getElementById("addJobBtn");
const generateBtn = document.getElementById("generateBtn");
const errorEl = document.getElementById("error");

const genCountEl = document.getElementById("genCount");
const costTotalEl = document.getElementById("costTotal");

let genCount = 0;

// ---- Startup: is the API key configured? ----
async function init() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.configured) showApp();
    else setupEl.classList.remove("hidden");
  } catch {
    setupEl.classList.remove("hidden");
  }
}

function showApp() {
  setupEl.classList.add("hidden");
  appEl.classList.remove("hidden");
  if (!jobsEl.children.length) addJob(); // start with one
}

// ---- Save API key ----
saveKeyBtn.addEventListener("click", async () => {
  const key = apiKeyInput.value.trim();
  setupError.textContent = "";
  if (!key) {
    setupError.textContent = "Please enter your API key.";
    return;
  }
  saveKeyBtn.disabled = true;
  saveKeyBtn.textContent = "Saving…";
  try {
    const res = await fetch("/api/save-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not save key.");
    showApp();
  } catch (err) {
    setupError.textContent = err.message;
  } finally {
    saveKeyBtn.disabled = false;
    saveKeyBtn.textContent = "Save & Continue";
  }
});
apiKeyInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveKeyBtn.click();
});

// ---- Job management ----
const jobs = []; // list of job controller objects

function renumber() {
  [...jobsEl.children].forEach((card, i) => {
    card.querySelector(".job-title").textContent = `Image ${i + 1}`;
    // hide the remove button when only one job remains
    card.querySelector(".remove-job").style.display =
      jobsEl.children.length > 1 ? "" : "none";
  });
}

function addJob() {
  const card = jobTemplate.content.firstElementChild.cloneNode(true);
  const job = setupJob(card);
  jobs.push(job);
  jobsEl.appendChild(card);
  renumber();
  return job;
}

function setupJob(card) {
  const dropZone = card.querySelector(".drop-zone");
  const fileInput = card.querySelector(".file-input");
  const dropPrompt = card.querySelector(".drop-prompt");
  const sourcePreview = card.querySelector(".source-preview");
  const promptInput = card.querySelector(".prompt-input");
  const presetSelect = card.querySelector(".preset-select");
  const spinner = card.querySelector(".spinner");
  const resultPlaceholder = card.querySelector(".result-placeholder");
  const resultImage = card.querySelector(".result-image");
  const downloadBtn = card.querySelector(".download-btn");
  const jobError = card.querySelector(".job-error");
  const removeBtn = card.querySelector(".remove-job");

  const job = { card, file: null, busy: false };

  function setFile(file) {
    jobError.textContent = "";
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      jobError.textContent = "Please choose an image file.";
      return;
    }
    if (file.size > MAX_BYTES) {
      jobError.textContent = "Image is too large. Maximum size is 20MB.";
      return;
    }
    job.file = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      sourcePreview.src = e.target.result;
      sourcePreview.classList.remove("hidden");
      dropPrompt.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    })
  );
  dropZone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // Preset dropdown fills the prompt box; "Custom" leaves it for typing.
  presetSelect.addEventListener("change", () => {
    const preset = PRESETS[presetSelect.value];
    if (preset) promptInput.value = preset;
  });
  // If the user edits the text after picking a preset, mark it as custom.
  promptInput.addEventListener("input", () => {
    if (presetSelect.value && promptInput.value !== PRESETS[presetSelect.value]) {
      presetSelect.value = "";
    }
  });

  removeBtn.addEventListener("click", () => {
    const i = jobs.indexOf(job);
    if (i !== -1) jobs.splice(i, 1);
    card.remove();
    renumber();
  });

  // Returns a promise; resolves true if a generation succeeded.
  job.run = async function run() {
    jobError.textContent = "";
    if (!job.file) {
      jobError.textContent = "Upload an image first.";
      return false;
    }
    if (!promptInput.value.trim()) {
      jobError.textContent = "Enter a prompt first.";
      return false;
    }

    job.busy = true;
    spinner.classList.remove("hidden");
    resultImage.classList.add("hidden");
    downloadBtn.classList.add("hidden");
    resultPlaceholder.classList.add("hidden");

    const form = new FormData();
    form.append("image", job.file);
    form.append("prompt", promptInput.value.trim());

    try {
      const res = await fetch("/api/generate", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Generation failed.");

      resultImage.src = data.image;
      resultImage.classList.remove("hidden");
      downloadBtn.href = data.download_url;
      downloadBtn.setAttribute("download", data.filename || "result.png");
      downloadBtn.classList.remove("hidden");
      return true;
    } catch (err) {
      jobError.textContent = err.message;
      resultPlaceholder.classList.remove("hidden");
      return false;
    } finally {
      job.busy = false;
      spinner.classList.add("hidden");
    }
  };

  return job;
}

addJobBtn.addEventListener("click", addJob);

// ---- Generate all jobs in parallel ----
generateBtn.addEventListener("click", async () => {
  errorEl.textContent = "";
  if (!jobs.length) return;

  generateBtn.disabled = true;
  addJobBtn.disabled = true;
  generateBtn.textContent = "Generating…";

  // Run jobs through a concurrency pool so we don't overload the model.
  const queue = [...jobs];
  let succeeded = 0;
  async function worker() {
    while (queue.length) {
      const job = queue.shift();
      const ok = await job.run();
      if (ok) succeeded += 1;
    }
  }
  const workers = Array.from(
    { length: Math.min(MAX_CONCURRENT, jobs.length) },
    worker
  );
  await Promise.all(workers);

  genCount += succeeded;
  genCountEl.textContent = genCount;
  costTotalEl.textContent = (genCount * COST_PER_IMAGE).toFixed(2);

  generateBtn.disabled = false;
  addJobBtn.disabled = false;
  generateBtn.textContent = "Generate all";
});

init();
