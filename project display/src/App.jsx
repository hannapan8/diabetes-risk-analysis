import React from "react";
import "./index.css";

const HERO_LINKS = [
  {
    label: "View Project on GitHub",
    href: "https://github.com/hannapan8/diabetes-risk-analysis",
    variant: "primary",
  },
  {
    label: "Open Analysis Notebook",
    href: "/notebooks/eda.ipynb",
    variant: "secondary",
  },
  {
    label: "Read Full Report",
    href: "/report.pdf",
    variant: "secondary",
  },
];

const FIGURES = [
  {
    src: "/figures/diabetes_age.png",
    alt: "Bar chart of diabetes prevalence by age group",
    caption:
      "Diabetes prevalence by age group. Risk increases as one ages.",
  },
  {
    src: "/figures/diabetes_bmi.png",
    alt: "Box plot of diabetes prevalence by BMI category",
    caption:
      "Diabetes prevalence by BMI category. People with diabetes report to have slightly higher BMIs.",
  },
  {
    src: "/figures/diabetes_education.png",
    alt: "Bar chart of diabetes prevalence by education level",
    caption:
      "There seems to be a pattern as diabetes prevalence decreases as college education level increases.",
  },
  {
    src: "/figures/diabetes_healthcare.png",
    alt: "Bar chart of diabetes prevalence by whether or not someone has healthcare",
    caption:
      "People who have healthcare are more likely to be diagnosed.",
  },
  {
    src: "/figures/diabetes_income.png",
    alt: "Bar chart of diabetes prevalence by income levels",
    caption:
      "Those with higher income are less likely to develop diabetes and vice versa.",
  },
  {
    src: "/figures/diabetes_lifestyle.png",
    alt: "Bar chart of diabetes prevalence by lifestyle factor",
    caption:
      "Lifestyle factors like physical activities and eating/drinking healthy have an effect on whether or not someone develops diabetes.",
  },
  {
    src: "/figures/diabetes_sex.png",
    alt: "Bar chart of diabetes prevalence by an individual's sex (male or female)",
    caption:
      "Men appear to be more likely to develop diabetes in this study.",
  },
];

function Pill({ children }) {
  return <span className="pill">{children}</span>;
}

function HeroLink({ href, label, variant }) {
  return (
    <a
      className={`hero-link ${variant === "secondary" ? "secondary" : ""}`}
      href={href}
      target="_blank"
      rel="noreferrer"
    >
      {label}
    </a>
  );
}

function FigureCard({ src, alt, caption }) {
  return (
    <figure className="figure-card">
      <div className="figure-media">
        <img src={src} alt={alt} loading="lazy" />
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export default function App() {
  return (
    <div className="app-root">
      <div className="background-orbit orbit-1" />
      <div className="background-orbit orbit-2" />
      <main className="page">
        <header className="hero">
          <div className="hero-text">
            <p className="eyebrow">Data Visualization Portfolio · Case Study</p>
            <h1>Predicting Diabetes Risk from Lifestyle &amp; Demographics</h1>
            <p className="hero-subtitle">
              A data visualization and modeling project using the CDC Diabetes
              Health Indicators dataset (~200k records) to explain which factors are most strongly associated with
              diabetes risk. Tech stack: Python, React, JavaScript, Jupyter.
            </p>
            <div className="pill-row">
              <Pill>Data Storytelling</Pill>
              <Pill>Python · pandas · scikit-learn</Pill>
              <Pill>matplotlib · seaborn</Pill>
              <Pill>Front-end Visualization</Pill>
            </div>
            <div className="hero-links">
              {HERO_LINKS.map((link) => (
                <HeroLink key={link.label} {...link} />
              ))}
            </div>
          </div>
        </header>

        <section className="grid grid-two">
          <article className="card">
            <h2>What I Explored</h2>
            <p>
              The core question:{" "}
              <strong>
                How do age, BMI, and everyday behaviors combine to shape the
                likelihood of being diagnosed with diabetes?
              </strong>
            </p>
            <ul>
              <li>
                Cleaned and subset a large public health dataset (~200k rows).
              </li>
              <li>
                Visualized distributions and class imbalance for diabetic vs.
                non-diabetic groups.
              </li>
              <li>
                Analyzed how risk changes across age bands, BMI categories, and
                income levels.
              </li>
              <li>
                Built logistic regression and random forest models to quantify
                risk.
              </li>
            </ul>
            <p>
              The emphasis throughout is on{" "}
              <strong>clear, readable visuals</strong> that each answer a
              specific question, rather than standalone plots.
            </p>
          </article>

          <article className="card">
            <h2>How I Told the Story</h2>
            <p>
              I structured the project as a narrative that moves from raw data
              to an interface-ready summary:
            </p>
            <ul>
              <li>
                Exploratory charts that reveal who is most at risk in different
                subgroups.
              </li>
              <li>
                Model-based visualizations (coefficients, feature importances)
                to show which variables matter most.
              </li>
              <li>
                A layout that can be turned into a React dashboard to surface
                key metrics and charts in a product context.
              </li>
            </ul>
            <p>
              This page is a compact view of that pipeline:{" "}
              <strong>data → visuals → interpretable insights</strong>.
            </p>
          </article>
        </section>

        <section className="card gallery-card">
          <div className="section-header">
            <h2>Selected Visualizations</h2>
            <p>
              A sample of the visuals I created to explain patterns in the
              dataset. Each figure is paired with interpretation to emphasize
              storytelling, not just plotting.
            </p>
          </div>
          <div className="gallery-grid">
            {FIGURES.map((fig) => (
              <FigureCard key={fig.src} {...fig} />
            ))}
          </div>
        </section>

        <section className="card summary-card">
          <h2>Summary</h2>
          <p>
            This project combines{" "}
            <strong>data cleaning</strong>,{" "}
            <strong>visual exploration</strong>, and{" "}
            <strong>interpretable modeling</strong> to make chronic disease risk
            easier to understand. The same patterns and layouts could be
            embedded into a product-grade dashboard to support public health or
            clinical decision-making.
          </p>
          <p className="signature">
            Built by <strong>Hanna Pan</strong> · Informatics &amp; ACMS (Data
            Science &amp; Statistics), University of Washington.
          </p>
        </section>
      </main>
    </div>
  );
}
