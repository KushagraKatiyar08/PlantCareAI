# Repository Guidelines

Welcome to **PlantCareAI**. This guide provides the necessary standards and workflows for contributors and AI agents to maintain consistency and quality across the codebase.

## Project Structure & Module Organization
The repository is organized to separate core logic from configuration and assets:
- `/src`: Contains all primary source code.
- `/tests`: Unit and integration tests.
- `/docs`: Project documentation and API references.
- `/assets`: Static files, images, and plant database schemas.

## Build, Test, and Development Commands
Use the following commands for local development:
- `npm install`: Install project dependencies.
- `npm run dev`: Start the development server with hot-reloading.
- `npm test`: Execute the full test suite.
- `npm run build`: Compile the project for production.

## Coding Style & Naming Conventions
To ensure readability, follow these standards:
- **Indentation**: 2 spaces.
- **Naming**: 
  - `camelCase` for variables and functions.
  - `PascalCase` for classes and components.
  - `kebab-case` for file names (e.g., `plant-analyzer.js`).
- **Linting**: Use ESLint and Prettier for automatic formatting.

## Testing Guidelines
We strive for high reliability in plant diagnostics:
- **Framework**: Jest is used for all testing.
- **Naming**: Test files must end in `.test.js` or `.spec.js`.
- **Coverage**: New features should include unit tests covering the main success and failure paths.

## Commit & Pull Request Guidelines
- **Commits**: Use conventional commits (e.g., `feat: add watering reminder`, `fix: resolve leaf-spot detection bug`).
- **Pull Requests**: 
  - Provide a clear description of the changes.
  - Link the relevant issue number.
  - Include screenshots for any UI changes.
  - Ensure all tests pass before requesting a review.