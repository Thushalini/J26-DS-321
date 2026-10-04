# Buddy.ds

The in-app AI agent is called **Buddy**.

## Tech stack

- Next.js 16 (App Router) + React 19 + TypeScript
- Tailwind CSS v4

## Getting started

You need Node.js 20.9 or newer.

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

| Command             | What it does                            |
| ------------------- | --------------------------------------- |
| `npm run dev`       | Start the dev server                    |
| `npm run build`     | Production build                        |
| `npm run start`     | Run the production build                |
| `npm run typecheck` | Check TypeScript types without building |

## Project structure

```
app/
  layout.tsx      Root layout and page metadata
  page.tsx        Home page (placeholder)
  globals.css     Tailwind setup and global styles
components/       Reusable UI components
lib/              Helpers, types and data access
public/           Static files (images, icons)
```

Imports can use the `@/` alias for the project root, e.g. `@/components/...`.

## Working as a team

- **Never commit straight to `main`.** Create a branch per task: `feature/<short-name>` or `fix/<short-name>`.
- Open a pull request and get one teammate to review before merging.
- Pull `main` before starting new work.
- Run `npm run build` before opening a pull request.
- Environment variables: copy `.env.example` to `.env.local`. Never commit `.env.local`.
