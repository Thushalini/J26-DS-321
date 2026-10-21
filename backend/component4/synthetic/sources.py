"""Sources behind the synthetic generator's rules. "checked" turns True only after the student has opened the
source and confirmed the stated location says what "supports" claims; until then results stay provisional."""

SOURCES = {
    "nenni2025": {
        "cite": "Nenni et al. (2025). How artificial intelligence will transform project management in the age of "
                "digitization: a systematic literature review. Management Review Quarterly 75, 1669-1716.",
        "where": "Introduction (quotes the Standish Group)",
        "supports": "About 35% of projects succeed: success share 0.35 (secondary citation of Standish).",
        "checked": False,
    },
    "chaos2020": {
        "cite": "The Standish Group (2020). CHAOS 2020: Beyond Infinity. Paywalled; figures via secondary summaries.",
        "where": "Headline figures (31% successful, 50% challenged, 19% failed); success = on time, on budget, full scope",
        "supports": "Failed share 0.19 (3-class only); success defined by schedule; about half of projects run late.",
        "checked": False,
    },
    "prause2026": {
        "cite": "Prause, M. (2026). The Machine Learning Canvas: Empirical findings on why strategy matters more than "
                "AI code generation. arXiv:2601.01839 (published as 'Determinants of Machine Learning Project Success', "
                "Systems 14(8), 1018).",
        "where": "Table II (success items), Table VII (path estimates), Section V (sample, 14% real business impact)",
        "supports": "Success = performance goals, business objectives, stakeholder satisfaction; Success<-Ecosystem "
                    "0.202 (p=.012), Success<-Strategy 0.163 (p=.043); most projects experimental.",
        "checked": False,
    },
    "rand2024": {
        "cite": "Ryseff, J., De Bruhl, B. F. & Newberry, S. J. (2024). The root causes of failure for artificial "
                "intelligence projects and how they can succeed. RAND RR-A2680-1. Cited via prause2026 [6].",
        "where": "prause2026, Section I: over 80% of AI projects fail, twice IT; five failure modes",
        "supports": "Poor data quality and problems beyond current AI capability lead to failure.",
        "checked": False,
    },
    "zowghi2002": {
        "cite": "Zowghi, D. & Nurmuliani, N. (2002). A study of the impact of requirements volatility on software "
                "project performance. APSEC 2002, 3-11.",
        "where": "Abstract and results",
        "supports": "Requirements volatility significantly contributes to schedule and cost overrun: drift lowers success.",
        "checked": False,
    },
    "sambasivan2021": {
        "cite": "Sambasivan, N. et al. (2021). 'Everyone wants to do the model work, not the data work': Data cascades "
                "in high-stakes AI. CHI 2021. (Check the full author list on the paper.)",
        "where": "Abstract (53 practitioners; data cascades 92% prevalence)",
        "supports": "Data-quality problems are common and cause downstream failures: data quality raises success.",
        "checked": False,
    },
    "gundersen2018": {
        "cite": "Gundersen, O. E. & Kjensmo, S. (2018). State of the art: Reproducibility in artificial intelligence. "
                "AAAI 32(1).",
        "where": "Abstract (400 IJCAI/AAAI papers; 20-30% of variables documented per factor)",
        "supports": "Reproducibility documentation is usually low: reproducibility skewed low (mean about 0.25).",
        "checked": False,
    },
    "breck2017": {
        "cite": "Breck, E., Cai, S., Nielsen, E., Salib, M. & Sculley, D. (2017). The ML Test Score: A rubric for ML "
                "production readiness and technical debt reduction. IEEE Big Data 2017.",
        "where": "Section I (no test implemented by more than 80% of teams), Section VI-A and Table V, Section VII (36 teams)",
        "supports": "Equal points per test; even professional teams implement only part of the rubric: readiness skewed low.",
        "checked": False,
    },
    "serban2020": {
        "cite": "Serban, A., van der Blom, K., Hoos, H. & Visser, J. (2020). Adoption and effects of software "
                "engineering best practices in machine learning. ESEM 2020.",
        "where": "Abstract (29 practices, 313 practitioners)",
        "supports": "Adopting engineering practices predicts perceived quality, agility and traceability: readiness raises success.",
        "checked": False,
    },
    "kopec2023": {
        "cite": "Kopec-Harding, K., Eraslan, S., Cai, B., Embury, S. M. & Jay, C. (2023). The impact of unequal "
                "contributions in student software engineering team projects. Journal of Systems and Software 206, 111839.",
        "where": "Abstract",
        "supports": "No consistent association between contribution inequality and performance (small weight); at least "
                    "half the teams had low or extreme inequality (balance skewed below the middle).",
        "checked": False,
    },
    "martinez2021": {
        "cite": "Martinez, I., Viles, E. & Olaizola, I. G. (2021). A survey study of success factors in data science "
                "projects. IEEE (arXiv:2201.06310).",
        "where": "Section IV-C and Fig. 3 (237 professionals)",
        "supports": "Team collaboration and coordination is among the top three perceived success factors.",
        "checked": False,
    },
    "wang2017": {
        "cite": "Wang, J., Veugelers, R. & Stephan, P. (2017). Bias against novelty in science: A cautionary tale for "
                "users of bibliometric indicators. Research Policy 46(8), 1416-1436.",
        "where": "Abstract",
        "supports": "Novel work is high-risk, high-reward (more uncertainty, later recognition): no clear direction, "
                    "so novelty is neutral (the student's decision).",
        "checked": False,
    },
    "forouzeshnejad2025": {
        "cite": "ForouzeshNejad et al. (2025). Data-driven predictive modelling of agile projects using explainable "
                "artificial intelligence. Electronics 14(13), 2609.",
        "where": "Association-rule (Apriori) results",
        "supports": "Technical debt and dependency problems go with failure: experiment issues lower success.",
        "checked": False,
    },
}
