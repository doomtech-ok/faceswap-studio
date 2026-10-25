<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Project rules

- Face-swap UI is frontend-only for now: uploads stay in the browser as object URLs and the "Start" action is a placeholder; why: the user asked for no backend or database yet.
- Face-swap UI pieces live in `src/components/face-swap/`; why: keeps the single-page tool's parts together.
- The app is dark-only (tokens defined in `:root`, no `.dark` toggle); why: the requested design is a dark theme.
