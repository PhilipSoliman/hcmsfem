# Contributing to HCMSFEM

## Commit Signing Requirements

All commits to this repository must be signed and verified. This ensures the authenticity and integrity of contributions.

### Setting up commit signing:

1. **Generate an SSH key** (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Configure git to use SSH signing**:
   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global commit.gpgsign true
   ```

3. **Add your SSH key to GitHub**:
   - Go to GitHub Settings → SSH and GPG keys
   - Add your public key for both "Authentication" and "Signing"

4. **Verify your setup**:
   Make a test commit and check that it shows as "Verified" on GitHub.

### Development Guidelines

- Ensure all commits are signed and verified
- Follow the existing code style and structure
- Add appropriate tests for new functionality
- Update documentation as needed

For questions about contributing, please open an issue.
