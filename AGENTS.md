# Repository Guidelines

## Project Structure

This repository is newly initialized and currently contains no application source, tests, or assets. Keep production code, tests, and supporting resources in clearly named directories as the project takes shape (for example, `src/`, `tests/`, and `assets/`). Update this guide when the structure or technology stack is established.

## Build, Test, and Development

No build system or development commands are configured yet. When adding one, document the canonical setup, run, and test commands here and keep them reproducible from the repository root.

## Coding Style and Naming

Follow the conventions of the chosen language and framework. Use consistent formatting, descriptive names, and focused modules. Add the formatter or linter configuration alongside the code it governs, and run it before submitting changes.

## Testing Guidelines

No test framework or coverage requirement is configured. Add tests with the project’s selected framework; place them in `tests/` or beside the code under test, and use clear names that describe the behavior being checked. Document the command contributors should run.

## Commits and Pull Requests

There is no commit history yet, so repository-specific message conventions are not established. Write concise, imperative commit subjects (for example, `Add document search endpoint`). Pull requests should explain the change and its motivation, list validation performed, and link relevant issues. Include screenshots for user-facing visual changes.

## Security and Configuration

Do not commit credentials, tokens, or machine-specific secrets. Keep local configuration in ignored environment files and provide safe examples (such as `.env.example`) when configuration is introduced.
