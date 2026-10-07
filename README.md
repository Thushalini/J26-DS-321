# Buddy.ds

The in-app AI agent is called **Buddy**.

## Repository layout

```
frontend/    Next.js web app (the screens)
backend/     Python backend (the four components)
```

The frontend asks the backend for data over HTTP; the backend runs the models and returns the results.

## Frontend

Next.js 16 (App Router) + React 19 + TypeScript + Tailwind CSS v4. You need Node.js 20.9 or newer.

```bash
cd frontend
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

```
frontend/
  app/
    layout.tsx      Root layout and page metadata
    page.tsx        Home page (placeholder)
    globals.css     Tailwind setup and global styles
  components/       Reusable UI components
  lib/              Helpers, types and API calls
  public/           Static files (images, icons)
```

Imports can use the `@/` alias for the `frontend/` folder, e.g. `@/components/...`.

## Backend

Python, one folder per component. See [backend/README.md](backend/README.md) for setup.

## Working as a team

- **Never commit straight to `main`.** Create a branch per task: `feature/<short-name>` or `fix/<short-name>`.
- Open a pull request and get one teammate to review before merging.
- Pull `main` before starting new work.
- Run `npm run build` in `frontend/` before opening a pull request that touches the frontend.
- Environment variables: copy `.env.example` to `.env.local` (frontend) or `.env` (backend). Never commit the real files.
