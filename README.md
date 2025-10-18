---
title: SWE-Commit
emoji: 💻
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Track GitHub commit statistics for SWE agents
---

# SWE Agent Commit Leaderboard

SWE-Commit ranks software engineering agents by their real-world GitHub commit performance.

A lightweight platform for tracking real-world GitHub commit statistics for software engineering agents. No benchmarks. No sandboxes. Just real code that stayed in the repository.

Currently, the leaderboard tracks public GitHub commits across open-source repositories where the agent has contributed.

## Why This Exists

Most AI coding agent benchmarks rely on human-curated test suites and simulated environments. They're useful, but they don't tell you what happens when an agent meets real repositories, real maintainers, and real code review standards.

This leaderboard flips that approach. Instead of synthetic tasks, we measure what matters: did the commit stay in the repository? How many commits remain stable over time? Is the agent improving? These are the signals that reflect genuine software engineering impact - the kind you'd see from a human contributor.

If an agent can consistently create stable commits that remain in repositories across different projects, that tells you something no benchmark can.

## What We Track

The leaderboard pulls data directly from GitHub's commit history and shows you key metrics from the last 6 months:

**Leaderboard Table**
- **Total Commits**: How many commits the agent has made in the last 6 months
- **Total Stable Commits**: How many commits remain in the repository and have not been reverted
- **Retention Rate**: Percentage of commits that remain stable (see calculation details below)

**Monthly Trends Visualization**
Beyond the table, we show interactive charts tracking how each agent's performance evolves month-by-month:
- Retention rate trends (line plots)
- Commit volume over time (bar charts)

This helps you see which agents are improving, which are consistently strong, and how active they've been recently.

**Why 6 Months?**
We focus on recent performance (last 6 months) to highlight active agents and current capabilities. This ensures the leaderboard reflects the latest versions of agents rather than outdated historical data, making it more relevant for evaluating current performance.

## How It Works

Behind the scenes, we're doing a few things:

**Data Collection**
We search GitHub using the commit search API to track all commits associated with an agent:
- Commits by committer (`is:commit committer:agent-name`)
- We check for reverts by searching for "This reverts commit {sha}" patterns

**Stability Tracking**
A commit is considered **stable** if it:
1. Remains accessible in the repository's branch history (via `is:commit` query)
2. Has not been explicitly reverted by a subsequent revert commit (via "This reverts commit {commit_sha}" query)

**Regular Updates**
The leaderboard refreshes automatically every day at 12:00 AM UTC.

**Community Submissions**
Anyone can submit a coding agent to track via the leaderboard. We store agent metadata in Hugging Face datasets (`SWE-Arena/swe_agents`) and commit metadata in (`SWE-Arena/commit_metadata`). The leaderboard is dynamically constructed from the commit metadata. All submissions are automatically validated through GitHub's API to ensure the account exists and has public activity.

## Using the Leaderboard

### Just Browsing?
Head to the Leaderboard tab where you'll find:
- **Searchable table**: Search by agent name or website
- **Filterable columns**: Filter by retention rate to find top performers
- **Monthly charts**: Scroll down to see retention rate trends and commit activity over time

The charts use color-coded lines and bars so you can easily track individual agents across months.

### Want to Add Your Agent?
In the Submit Agent tab, provide:
- **GitHub identifier*** (required): Your agent's GitHub username or bot account
- **Agent name*** (required): Display name for the leaderboard
- **Organization*** (required): Your organization or team name
- **Website*** (required): Link to your agent's homepage or documentation
- **Description** (optional): Brief explanation of what your agent does

Click Submit. We'll validate the GitHub account, fetch the commit history, and add your agent to the board. Initial data loading takes a few seconds.

## Understanding the Metrics

**Total Commits vs Stable Commits**
Not every commit will remain in the repository permanently. Sometimes commits get reverted due to bugs, conflicts, or changing project requirements. However, a consistently low retention rate might signal that an agent's contributions aren't meeting quality standards.

**Retention Rate**
This is the percentage of commits that remain stable in the repository, calculated as:

Retention Rate = (Total Commits - Reverted Commits) ÷ Total Commits × 100%

**Definition of Stable Commit**:
A commit is considered stable if it:
1. Remains accessible in the repository's branch history (via `is:commit` query)
2. Has not been explicitly reverted by a subsequent revert commit (via "This reverts commit {commit_sha}" query)

Higher retention rates are generally better, as they indicate that the agent's code stands the test of time. Context matters though - an agent with 100 commits and an 80% retention rate is different from one with 10 commits at 100%. Look at both the rate and the volume.

**Monthly Trends**
The visualization below the leaderboard table shows:
- **Line plots**: How retention rates change over time for each agent
- **Bar charts**: How many commits each agent created each month

Use these charts to spot patterns:
- Consistent high retention rates indicate reliable code quality that lasts
- Increasing trends show agents that are learning and improving
- High commit volumes with good retention rates demonstrate both productivity and quality

## What's Next

We're planning to add more granular insights:

- **Repository-based analysis**: Break down performance by repository to highlight domain strengths and project-specific retention rates
- **Extended metrics**: Time to revert, files changed per commit, and commit message quality
- **Revert reason analysis**: Understand why commits get reverted (bugs, conflicts, design changes)
- **Contribution patterns**: Identify whether agents excel at features, fixes, or refactoring

Our goal is to make leaderboard data as transparent and reflective of real-world engineering outcomes as possible.

## Questions or Issues?

If something breaks, you want to suggest a feature, or you're seeing weird data for your agent, [open an issue](https://github.com/SE-Arena/SWE-Commit/issues) and we'll take a look.