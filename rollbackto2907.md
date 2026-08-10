# Rollback to v2026-07-29-experiment

Tag `v2026-07-29-experiment` is a frozen snapshot of `main` as of the last
commit on 2026-07-29 (the day of the successful flight test). Use this if a
newer version needs to be rolled back before/during a flight test.

## 1. Protect current work first

```
git stash push -u -m "pre-rollback WIP"
```

`-u` includes untracked files. Recover anytime with `git stash pop`.

## 2. Recommended: test the old version without touching `main`

```
git checkout -b flight-rollback v2026-07-29-experiment
```

Puts the exact 2026-07-29 code in the working tree on its own branch. Run/fly
with it. `main` stays untouched, so this is zero-risk while just validating.

## 3. If confirmed good and `main` should permanently go back to it

```
git checkout main
git reset --hard v2026-07-29-experiment
```

Note: if `origin/main` is ahead of the tag (newer commits pushed), this only
rewinds the local `main`. Pushing the rollback to GitHub afterwards requires
`git push --force`, which rewrites shared history — do this deliberately and
not automatically.

## 4. Quick option: just run the old version now, no branch

```
git checkout v2026-07-29-experiment
```

Detached HEAD at that exact commit. Fine for running/testing; don't commit
new work directly here (easy to lose) — branch off (step 2) first if changes
are needed.

## Recommended flight-day procedure

Under time pressure, use step 2 (`flight-rollback` branch): fastest way to
get the known-good code running, no risk to `main`, decide about `main`/push
afterward with a clear head.
