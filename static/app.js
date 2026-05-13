(() => {
  const form = document.getElementById("matchup-form");
  const inputA = document.getElementById("fighter-a");
  const inputB = document.getElementById("fighter-b");
  const sugA = document.getElementById("suggestions-a");
  const sugB = document.getElementById("suggestions-b");
  const submitBtn = document.getElementById("submit-btn");
  const loading = document.getElementById("loading");
  const errorBanner = document.getElementById("error");
  const results = document.getElementById("results");
  const profileAEl = document.getElementById("profile-a");
  const profileBEl = document.getElementById("profile-b");
  const matchupEl = document.getElementById("matchup");

  function debounce(fn, ms) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }

  function attachAutocomplete(input, list) {
    const fetchSuggestions = debounce(async () => {
      const q = input.value.trim();
      if (q.length < 2) { list.hidden = true; return; }
      try {
        const res = await fetch(`/api/fighters?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        list.innerHTML = "";
        if (!data.suggestions || data.suggestions.length === 0) {
          list.hidden = true;
          return;
        }
        for (const name of data.suggestions) {
          const li = document.createElement("li");
          li.textContent = name;
          li.addEventListener("mousedown", (e) => {
            e.preventDefault();
            input.value = name;
            list.hidden = true;
          });
          list.appendChild(li);
        }
        list.hidden = false;
      } catch (e) { list.hidden = true; }
    }, 180);

    input.addEventListener("input", fetchSuggestions);
    input.addEventListener("focus", fetchSuggestions);
    input.addEventListener("blur", () => setTimeout(() => { list.hidden = true; }, 100));
  }

  attachAutocomplete(inputA, sugA);
  attachAutocomplete(inputB, sugB);

  function showError(message, missing) {
    errorBanner.innerHTML = "";
    const p = document.createElement("p");
    p.textContent = message;
    errorBanner.appendChild(p);
    if (missing && missing.length) {
      for (const m of missing) {
        const note = document.createElement("p");
        note.style.marginTop = "0.5rem";
        note.innerHTML = `<strong>"${escapeHtml(m.input)}"</strong> not found.`;
        if (m.suggestions && m.suggestions.length) {
          note.innerHTML += " Did you mean:";
          const ul = document.createElement("ul");
          for (const s of m.suggestions) {
            const li = document.createElement("li");
            li.textContent = s;
            ul.appendChild(li);
          }
          note.appendChild(ul);
        }
        errorBanner.appendChild(note);
      }
    }
    errorBanner.hidden = false;
  }

  function clearError() { errorBanner.hidden = true; errorBanner.innerHTML = ""; }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function tierClass(tier) {
    return "tier-" + String(tier || "average").replace(/\s+/g, "-");
  }

  function renderProfile(el, profile) {
    el.innerHTML = `
      <h2>${escapeHtml(profile.name)}</h2>
      <p class="sub">${escapeHtml(profile.weight_class)} • ${escapeHtml(profile.record)}</p>
      <span class="style-tag">${escapeHtml(profile.style_label)}</span>
      <div class="grades">
        <div class="grade">
          <div class="label">Striking</div>
          <div class="value ${tierClass(profile.striking_grade)}">${escapeHtml(profile.striking_grade)}</div>
        </div>
        <div class="grade">
          <div class="label">Grappling</div>
          <div class="value ${tierClass(profile.grappling_grade)}">${escapeHtml(profile.grappling_grade)}</div>
        </div>
        <div class="grade">
          <div class="label">Cardio</div>
          <div class="value ${tierClass(profile.cardio_durability_grade)}">${escapeHtml(profile.cardio_durability_grade)}</div>
        </div>
      </div>
      <h3>Strengths</h3>
      <ul>${profile.top_strengths.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
      <h3>Weaknesses</h3>
      <ul>${profile.top_weaknesses.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
      ${profile.x_factors && profile.x_factors.length ? `
        <h3>X-factors</h3>
        <ul>${profile.x_factors.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
      ` : ""}
      <h3>Recent form</h3>
      <p class="recent-form">${escapeHtml(profile.recent_form)}</p>
    `;
  }

  function magClass(mag) { return "mag-" + String(mag || "even"); }

  function renderMatchup(m) {
    const advRows = m.advantages.map(a => `
      <div class="advantage-row">
        <div class="cat">${escapeHtml(a.category.replace(/_/g, " "))}</div>
        <div>${escapeHtml(a.favored_fighter)}</div>
        <div class="${magClass(a.magnitude)}">${escapeHtml(a.magnitude)}</div>
        <div>${escapeHtml(a.reasoning)}</div>
      </div>
    `).join("");

    matchupEl.innerHTML = `
      <h2>Matchup Breakdown</h2>
      ${advRows}
      <h3 style="margin-top:1.2rem;">Stylistic clash</h3>
      <p>${escapeHtml(m.stylistic_clash)}</p>
      <h3>Key factors</h3>
      <ul>${m.key_factors.map(f => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
      <div class="prediction">
        <div>Predicted winner: <span class="winner">${escapeHtml(m.predicted_winner)}</span>
          <span class="confidence">${escapeHtml(m.confidence)} confidence</span>
        </div>
        <p style="margin:0.6rem 0 0;">${escapeHtml(m.prediction_rationale)}</p>
      </div>
    `;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    results.hidden = true;
    submitBtn.disabled = true;
    loading.hidden = false;

    const payload = {
      fighter_a: inputA.value.trim(),
      fighter_b: inputB.value.trim(),
    };

    try {
      const res = await fetch("/api/matchup", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || `Request failed (${res.status})`, data.missing);
      } else {
        renderProfile(profileAEl, data.profile_a);
        renderProfile(profileBEl, data.profile_b);
        renderMatchup(data.matchup);
        results.hidden = false;
      }
    } catch (err) {
      showError(`Network error: ${err.message}`);
    } finally {
      loading.hidden = true;
      submitBtn.disabled = false;
    }
  });
})();
