import { useEffect, useState } from "react";
import "./KpkHero.css";

const STATS = [
  {
    value: "35",
    label: "Districts",
  },
  {
    value: "2023",
    label: "Census Data",
  },
  {
    value: "AI",
    label: "Opportunity Analysis",
  },
];

export default function KpkHero({ onExplore }) {
  const [activeStat, setActiveStat] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStat((current) => (current + 1) % STATS.length);
    }, 3000);

    return () => clearInterval(timer);
  }, []);

  return (
    <section className="business-hero">
      {/* Background */}
      <div className="business-hero-bg">
        <div className="business-grid" />
        <div className="business-glow glow-one" />
        <div className="business-glow glow-two" />

        <div className="kpk-map-lines">
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
      </div>

      {/* Navigation */}
      <nav className="business-nav">
        <div className="business-brand">
          <div className="brand-icon">AI</div>

          <div>
            <strong>AI Business Advisor</strong>
            <span>Khyber Pakhtunkhwa</span>
          </div>
        </div>

        <div className="business-nav-status">
          <span className="live-dot" />
          AI Analysis Online
        </div>
      </nav>

      {/* Hero content */}
      <div className="business-hero-content">

        <div className="business-eyebrow">
          <span className="eyebrow-dot" />
          KPK BUSINESS INTELLIGENCE
        </div>

        <h1>
          Business Opportunities
          <span>in Khyber Pakhtunkhwa</span>
        </h1>

        <p className="business-hero-description">
          Find the right business. In the right location.
          Our AI analyzes population, competition, budget,
          density and Census 2023 data to identify promising
          business opportunities across KPK.
        </p>

        {/* Buttons */}
        <div className="business-hero-actions">
          <button
            className="primary-business-btn"
            onClick={onExplore}
          >
            <span>Analyze a Location</span>
            <span className="btn-arrow">→</span>
          </button>

          <button
            className="secondary-business-btn"
            onClick={onExplore}
          >
            Explore KPK
          </button>
        </div>

        {/* Stats */}
        <div className="business-stats">
          {STATS.map((stat, index) => (
            <div
              key={stat.label}
              className={`business-stat ${
                activeStat === index ? "active" : ""
              }`}
            >
              <strong>{stat.value}</strong>
              <span>{stat.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* AI Analysis Card */}
      <div className="business-ai-card">

        <div className="ai-card-top">
          <div>
            <span className="ai-card-label">
              OPPORTUNITY ENGINE
            </span>

            <h3>AI Location Analysis</h3>
          </div>

          <div className="ai-status">
            <span />
            LIVE
          </div>
        </div>

        <div className="ai-card-map">

          <div className="map-ring ring-one" />
          <div className="map-ring ring-two" />
          <div className="map-ring ring-three" />

          <div className="map-center">
            <div className="map-pin">
              +
            </div>
          </div>

          <div className="map-point point-one">
            <span />
          </div>

          <div className="map-point point-two">
            <span />
          </div>

          <div className="map-point point-three">
            <span />
          </div>

        </div>

        <div className="ai-analysis-row">

          <div>
            <span>District Coverage</span>
            <strong>35 Districts</strong>
          </div>

          <div>
            <span>Data Source</span>
            <strong>Census 2023</strong>
          </div>

        </div>

        <div className="ai-card-footer">

          <div className="analysis-icon">
            ✦
          </div>

          <div>
            <strong>AI Opportunity Analysis</strong>
            <span>
              Population • Competition • Budget
            </span>
          </div>

        </div>

      </div>

      {/* Bottom scroll indicator */}
      <button
        className="hero-scroll"
        onClick={onExplore}
        aria-label="Scroll to business advisor"
      >
        <span>Explore opportunities</span>
        <div className="scroll-line">
          <span />
        </div>
      </button>

    </section>
  );
}