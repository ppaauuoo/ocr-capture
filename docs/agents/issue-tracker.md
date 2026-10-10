# Issue tracker: GitHub

Issues and specs for this repo live in GitHub Issues. Use the `gh` CLI.

## Conventions

- Create an issue with `gh issue create --title "..." --body "..."`.
- Read an issue and its comments with `gh issue view <number> --comments`.
- List issues with `gh issue list`, adding state and label filters as needed.
- Comment and manage labels with `gh issue comment` and `gh issue edit`.
- Use GitHub's native issue dependencies for blocking links. Add a dependency with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-database-id>`. The dependency uses the blocker's numeric database ID, not its issue number. If native dependencies are unavailable, put `Blocked by: #<number>` in the issue body.
- **PRs as a request surface: no.**

When a skill says to publish to the issue tracker, create a GitHub issue. When it asks to fetch a ticket, run `gh issue view <number> --comments`.
