# Security Policy

## Supported Versions

Security fixes are applied to the current development branch and the latest release only. Older releases are not maintained.

| Version | Supported |
| --- | --- |
| `main` branch | Yes |
| Latest release | Yes |
| Older releases | No |
| Forks and mirrors | No — report issues to the fork maintainer unless the issue also affects this repository |

## Scope

This policy applies to repositories maintained under the GitHub account `Manoj-2424150118`, including security tooling and research projects such as Aegis, Deceptive-Net, SIEM, FoldCrypt, Z-Security, and Wokie.

These projects are defensive, educational, and research-oriented tools. Some projects intentionally simulate vulnerable services, honeypots, decoy credentials, or attacker workflows for detection and research. Behaviour that is explicitly documented as simulation, deception, or honeypot functionality is not automatically a vulnerability. A report is in scope when it shows a real risk to users, maintainers, credentials, data, or systems beyond the documented simulation behaviour.

## Reporting a Vulnerability

Please do **not** open a public issue for a security vulnerability.

### Preferred method: GitHub Private Vulnerability Reporting

1. Go to the affected repository.
2. Open the **Security** tab.
3. Select **Report a vulnerability**.
4. Provide the details listed below.

### Fallback method: email

If Private Vulnerability Reporting is not available for the repository, email:

**sharmamanojkumar697@gmail.com**

Use this subject format:

```text
[SECURITY] <repository>: <short summary>
```

Do not send vulnerability reports by phone, social media, public issues, public pull requests, or public discussions.

## What to Include in a Report

A useful report includes:

- Repository name and affected component.
- Affected version, release, branch, or commit hash.
- Vulnerability type, such as authentication bypass, injection, unsafe deserialization, path traversal, cryptographic misuse, secret exposure, privilege escalation, or insecure storage.
- Clear description of the issue and the security impact.
- Step-by-step reproduction instructions.
- Proof of concept, sample input, request/response samples, screenshots, or logs where relevant.
- Environment details: operating system, runtime versions, deployment mode, and relevant configuration.
- Whether the issue is already public or known to others.
- Suggested fix or mitigation, if you have one.
- Your preferred contact method for follow-up and credit.

Redact real secrets, tokens, passwords, private keys, personal data, and third-party credentials from reports. If a proof of concept uses credentials, use test credentials created for that purpose.

## Response Timeline

This is a solo-maintainer project set, so timelines are realistic rather than corporate SLAs.

- **Acknowledgement:** within 5 business days.
- **Initial triage:** within 10 business days.
- **Status updates:** when there is material progress, a fix is available, or the assessment changes.
- **Fix timeline:** depends on severity, complexity, exploitability, and available maintainer time. Critical issues are prioritised, but no fixed remediation SLA is promised.

If you do not receive an acknowledgement within 5 business days, please send a short follow-up using the same private channel.

## Coordinated Disclosure

Please allow up to **90 days** from the initial report for investigation, remediation, and release before public disclosure.

Disclosure may be coordinated earlier if:

- A fix is released sooner.
- The issue is already widely public.
- There is evidence of active exploitation and users need immediate defensive guidance.

Extensions beyond 90 days may be requested for complex fixes, upstream dependency issues, or cases where an immediate release would create greater risk. Public disclosure should be factual, coordinated where possible, and should avoid publishing live secrets, personal data, or weaponised exploit material.

## Safe Harbor

Good-faith security research is welcome.

I will not pursue or support legal action against researchers who:

- Make a genuine effort to follow this policy.
- Only test systems, accounts, repositories, and data they own or are explicitly authorised to test.
- Avoid privacy violations, data destruction, service degradation, denial of service, spam, social engineering, and physical attacks.
- Do not access, modify, copy, or retain third-party data.
- Stop testing and report immediately if they encounter credentials, personal data, or production data belonging to others.
- Keep vulnerability details private until the coordinated disclosure process is complete.

If a researcher is unsure whether an action is permitted, ask privately before proceeding.

## Out of Scope

The following are generally out of scope unless a report demonstrates concrete security impact in this project:

- Automated scanner output without a working proof of concept.
- Missing HTTP security headers in local demos, examples, or documentation deployments.
- Self-XSS or issues requiring a user to paste attacker-controlled content into their own browser console.
- Social engineering, phishing, physical attacks, or attacks against the maintainer personally.
- Denial-of-service findings based only on high request volume.
- Reports against forks, mirrors, archived experiments, or third-party deployments not controlled by this maintainer.
- Intentionally generated decoy credentials, canary tokens, honeypot records, or simulated attacker data behaving as documented.
- Vulnerabilities only in third-party services, hosting platforms, or dependencies where there is no exploitable impact through this project.
- Best-practice suggestions without a demonstrated vulnerability.

## Credit

Reporters who follow coordinated disclosure may be credited in release notes or advisories unless they prefer to remain anonymous.
