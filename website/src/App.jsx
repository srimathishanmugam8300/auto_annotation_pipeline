import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";

const containerVariants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.55,
      ease: [0.22, 0.9, 0.3, 1],
      staggerChildren: 0.08
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.22, 0.9, 0.3, 1] }
  }
};

function cleanEventName(name) {
  return String(name || "")
    .replace(/^Event\s*-\s*/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

function EventCard({ eventData }) {
  const title = useMemo(() => cleanEventName(eventData.name), [eventData.name]);

  return (
    <motion.section
      className="event-card"
      variants={itemVariants}
      whileHover={{ y: -3 }}
      viewport={{ once: true, amount: 0.1 }}
    >
      <div className="event-head">
        <h2>{title}</h2>
      </div>

      <div className="event-body">
        <aside className="event-description-panel">
          <p className="event-description-title">Event Description</p>
          <div className="event-description-space" aria-label="Event description space" />
        </aside>

        <div className="event-gallery">
          {eventData.images.map((img, idx) => (
            <motion.figure
              key={`${img}-${idx}`}
              className="image-card"
              initial={{ opacity: 0, scale: 0.98, y: 10 }}
              whileInView={{ opacity: 1, scale: 1, y: 0 }}
              viewport={{ once: true, amount: 0.22 }}
              transition={{ duration: 0.45, delay: Math.min(idx * 0.04, 0.28) }}
            >
              <img src={img} alt="Event photo" loading="lazy" />
            </motion.figure>
          ))}
        </div>
      </div>
    </motion.section>
  );
}

export default function App() {
  const [events, setEvents] = useState([]);

  useEffect(() => {
    let isMounted = true;

    fetch("/events-data.json")
      .then((res) => res.json())
      .then((data) => {
        if (!isMounted || !Array.isArray(data)) return;
        setEvents(data);
      })
      .catch(() => {
        if (!isMounted) return;
        setEvents([]);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" aria-hidden="true" />
      <div className="ambient ambient-right" aria-hidden="true" />

      <motion.header
        className="page-header"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.65, ease: [0.22, 0.9, 0.3, 1] }}
      >
        <h1>Achievements</h1>
      </motion.header>

      <motion.main
        className="events-wrap"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {events.map((eventData, idx) => (
          <EventCard key={`${eventData.name}-${idx}`} eventData={eventData} />
        ))}
      </motion.main>
    </div>
  );
}
