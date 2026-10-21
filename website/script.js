const events = Array.isArray(window.EVENTS_DATA) ? window.EVENTS_DATA : [];
const eventsContainer = document.getElementById("eventsContainer");
const template = document.getElementById("eventTemplate");
const parallaxLayers = Array.from(document.querySelectorAll(".parallax"));
const orbs = Array.from(document.querySelectorAll(".bg-orb"));

let framePending = false;

function cleanEventName(name) {
  return String(name || "").replace(/^Event\s*-\s*/i, "").trim();
}

function createPhoto(path) {
  const fig = document.createElement("figure");
  fig.className = "photo reveal";

  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = path;
  img.alt = "Event photo";

  fig.append(img);
  return fig;
}

function createEventCard(eventData) {
  const fragment = template.content.cloneNode(true);
  const title = fragment.querySelector(".event-title");
  const gallery = fragment.querySelector(".gallery");

  title.textContent = cleanEventName(eventData.name);

  eventData.images.forEach((src) => {
    gallery.append(createPhoto(src));
  });

  return fragment;
}

function revealOnView() {
  const observer = new IntersectionObserver(
    (entries, ob) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in-view");
        ob.unobserve(entry.target);
      });
    },
    {
      threshold: 0.12,
      rootMargin: "0px 0px -8% 0px"
    }
  );

  document.querySelectorAll(".reveal").forEach((el, i) => {
    el.style.transitionDelay = `${Math.min(i % 6, 5) * 55}ms`;
    observer.observe(el);
  });
}

function applyParallax() {
  const y = window.scrollY || 0;

  parallaxLayers.forEach((el) => {
    const speed = Number(el.dataset.speed || 0);
    el.style.transform = `translate3d(0, ${Math.round(y * speed)}px, 0)`;
  });

  orbs.forEach((el, i) => {
    const speed = i === 0 ? 0.13 : -0.1;
    el.style.transform = `translate3d(0, ${Math.round(y * speed)}px, 0)`;
  });

  framePending = false;
}

function onScroll() {
  if (framePending) return;
  framePending = true;
  requestAnimationFrame(applyParallax);
}

if (!events.length) {
  const p = document.createElement("p");
  p.textContent = "No events found.";
  eventsContainer.append(p);
} else {
  events.forEach((eventData) => {
    eventsContainer.append(createEventCard(eventData));
  });

  revealOnView();
  applyParallax();
  window.addEventListener("scroll", onScroll, { passive: true });
}
