"""
data/tickets.py
────────────────────────────────────────────────────────────────────────────────
Curated dataset of IT support tickets organised by difficulty.

Each ticket is a dict with:
  - id          : unique identifier
  - text        : raw natural-language ticket body
  - label       : ground-truth {category, priority, route}
  - difficulty  : easy | medium | hard
  - notes       : human-readable explanation of why this label is correct
"""

TICKETS = [
    # ══════════════════════════════════════════════════════════════════════════
    # EASY  – clear single-issue, obvious keywords
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "E001",
        "text": (
            "My laptop screen is completely black and will not turn on. "
            "I have tried holding the power button for 30 seconds but nothing happens. "
            "I need this fixed urgently as I have a client presentation in 2 hours."
        ),
        "label": {"category": "hardware", "priority": "P1", "route": "hardware_team"},
        "difficulty": "easy",
        "notes": "Clear hardware failure + time-critical urgency → P1 hardware_team",
    },
    {
        "id": "E002",
        "text": (
            "I cannot log in to my corporate email account. "
            "It keeps saying 'incorrect password' even though I am sure I am typing it right. "
            "I have been locked out for 20 minutes."
        ),
        "label": {"category": "access", "priority": "P3", "route": "auth_team"},
        "difficulty": "easy",
        "notes": "Classic password lockout → access / P3 / auth_team",
    },
    {
        "id": "E003",
        "text": (
            "The office WiFi is completely down. None of the 50 employees on Floor 3 "
            "can connect to the internet. We cannot use any cloud tools."
        ),
        "label": {"category": "network", "priority": "P1", "route": "network_team"},
        "difficulty": "easy",
        "notes": "Floor-wide outage affecting 50 users → P1 network",
    },
    {
        "id": "E004",
        "text": (
            "Our inventory management application crashes every time I try to generate "
            "a monthly report. Error code: APP_EXCEPTION_0x44F. "
            "This is affecting end-of-month reconciliation."
        ),
        "label": {"category": "software", "priority": "P2", "route": "dev_team"},
        "difficulty": "easy",
        "notes": "Application crash with error code → software / P2 / dev_team",
    },
    {
        "id": "E005",
        "text": (
            "I received a suspicious email asking me to click a link and enter my "
            "corporate credentials. The sender domain looks fake. "
            "I did NOT click the link."
        ),
        "label": {"category": "security", "priority": "P2", "route": "security_team"},
        "difficulty": "easy",
        "notes": "Phishing attempt reported → security / P2 / security_team",
    },
    {
        "id": "E006",
        "text": (
            "My office printer is showing 'Paper Jam - Tray 2' and will not print anything. "
            "I have already removed all visible paper but the error persists."
        ),
        "label": {"category": "hardware", "priority": "P4", "route": "hardware_team"},
        "difficulty": "easy",
        "notes": "Physical printer issue → hardware / P4 low priority",
    },
    {
        "id": "E007",
        "text": (
            "Can you reset my VPN password? I forgot it and cannot connect to the "
            "company network while working from home."
        ),
        "label": {"category": "access", "priority": "P3", "route": "auth_team"},
        "difficulty": "easy",
        "notes": "Straightforward VPN credential reset → access / P3",
    },
    {
        "id": "E008",
        "text": (
            "The production database server is down. All customer-facing applications "
            "are returning 500 errors. Revenue impact is ongoing."
        ),
        "label": {"category": "software", "priority": "P1", "route": "dev_team"},
        "difficulty": "easy",
        "notes": "Production DB down with revenue impact → P1 / dev_team",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MEDIUM  – ambiguous, overlapping signals, incomplete info
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "M001",
        "text": (
            "Since this morning I cannot access the internal HR portal. "
            "My colleague on the same network can access it fine. "
            "I already restarted my computer."
        ),
        "label": {"category": "access", "priority": "P3", "route": "auth_team"},
        "difficulty": "medium",
        "notes": (
            "Ambiguous: could be network or access. Colleague works → not network outage. "
            "User-specific → access / auth_team. Not urgent → P3."
        ),
    },
    {
        "id": "M002",
        "text": (
            "My computer is running extremely slowly. Everything takes forever to load. "
            "I can still use it but it is almost unusable."
        ),
        "label": {"category": "hardware", "priority": "P3", "route": "hardware_team"},
        "difficulty": "medium",
        "notes": (
            "Slow PC: could be HW (RAM/disk) or SW (bloat/virus). "
            "No specific SW error cited → hardware is primary route. P3 partial impact."
        ),
    },
    {
        "id": "M003",
        "text": (
            "I think someone else is using my account. I noticed logins from a city "
            "I have never been to in the audit log. My password still works though."
        ),
        "label": {"category": "security", "priority": "P1", "route": "security_team"},
        "difficulty": "medium",
        "notes": (
            "Account compromise indicators → security team. "
            "Even though account still accessible, this is a P1 security event."
        ),
    },
    {
        "id": "M004",
        "text": (
            "The new software update pushed last night broke the shortcut keys in Excel. "
            "Some macros also stopped working. Not all users are affected."
        ),
        "label": {"category": "software", "priority": "P3", "route": "dev_team"},
        "difficulty": "medium",
        "notes": (
            "Post-update regression. Not full outage. Partial user impact → P3 software / dev_team."
        ),
    },
    {
        "id": "M005",
        "text": (
            "Our branch office in Mumbai cannot reach the main server. "
            "Local internet seems fine — they can browse the web. "
            "Only internal tools are unreachable."
        ),
        "label": {"category": "network", "priority": "P2", "route": "network_team"},
        "difficulty": "medium",
        "notes": (
            "VPN/WAN tunnel likely broken. Internet OK but internal unreachable → network. "
            "Whole branch impacted → P2."
        ),
    },
    {
        "id": "M006",
        "text": (
            "I keep getting 'Access Denied' when trying to open a shared folder on the "
            "file server. I had access last week. No one told me my permissions changed."
        ),
        "label": {"category": "access", "priority": "P3", "route": "auth_team"},
        "difficulty": "medium",
        "notes": "Permission revocation (accidental or policy-based) → access / auth_team / P3.",
    },
    {
        "id": "M007",
        "text": (
            "My keyboard randomly stops responding for a few seconds then comes back. "
            "I tried a different USB port. Same issue. Driver update did not help."
        ),
        "label": {"category": "hardware", "priority": "P4", "route": "hardware_team"},
        "difficulty": "medium",
        "notes": (
            "Intermittent keyboard issue: SW (driver) ruled out → hardware fault. "
            "Minor inconvenience → P4."
        ),
    },
    {
        "id": "M008",
        "text": (
            "Several users are reporting that emails sent to clients are bouncing back. "
            "Internal emails work fine. Started about 2 hours ago."
        ),
        "label": {"category": "network", "priority": "P2", "route": "network_team"},
        "difficulty": "medium",
        "notes": (
            "External email delivery failure → likely MX/DNS/relay issue → network. "
            "Business impact, multiple users → P2."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HARD  – multi-issue, cascading failures, ambiguous priority
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "H001",
        "text": (
            "URGENT: Our entire data centre is experiencing issues. "
            "The network switches are dropping packets intermittently, "
            "which is causing application timeouts across all services. "
            "Additionally, two physical servers have reported disk errors in RAID logs. "
            "Finance and HR systems are both down. This is a major business impact."
        ),
        "label": {"category": "network", "priority": "P1", "route": "network_team"},
        "difficulty": "hard",
        "notes": (
            "Multi-issue: network (primary/root cause) + hardware (disk). "
            "Network is root cause driving app failures. "
            "All systems down → P1. Primary route: network_team (escalate hardware separately)."
        ),
    },
    {
        "id": "H002",
        "text": (
            "We detected unusual outbound traffic from 3 workstations after a user "
            "accidentally ran an email attachment. Antivirus flagged it as malware. "
            "Those machines are now isolated but users cannot work. "
            "We also lost access to two shared drives."
        ),
        "label": {"category": "security", "priority": "P1", "route": "security_team"},
        "difficulty": "hard",
        "notes": (
            "Active malware incident → security is primary. "
            "Operational impact (users offline, drives lost) is secondary. "
            "P1 active breach scenario."
        ),
    },
    {
        "id": "H003",
        "text": (
            "Our CI/CD pipeline has been failing for 6 hours. "
            "Deployments are stuck. The build server ran out of disk space, "
            "and we think a misconfigured firewall rule is also blocking "
            "artifact uploads to the registry. Dev team is blocked completely."
        ),
        "label": {"category": "software", "priority": "P2", "route": "dev_team"},
        "difficulty": "hard",
        "notes": (
            "Pipeline failure: SW config (firewall rule + disk) → dev_team primary. "
            "Network team may need sub-ticket for firewall. "
            "Dev team blocked but prod not yet affected → P2."
        ),
    },
    {
        "id": "H004",
        "text": (
            "Multiple employees cannot login after last night's AD migration. "
            "Some users get 'Account Disabled', others see 'Domain not found'. "
            "Remote workers on VPN are completely locked out. "
            "On-site staff can log in sporadically. This is affecting ~200 users."
        ),
        "label": {"category": "access", "priority": "P1", "route": "auth_team"},
        "difficulty": "hard",
        "notes": (
            "Post-AD migration mass lockout → access. "
            "200 users affected including remote → P1. Auth team leads (network_team assists VPN)."
        ),
    },
    {
        "id": "H005",
        "text": (
            "Since the power outage this morning, half the office computers won't boot. "
            "The ones that do boot are getting BSOD errors. "
            "The network is also unstable — switches seem to have lost their configs. "
            "We cannot access any internal systems."
        ),
        "label": {"category": "hardware", "priority": "P1", "route": "hardware_team"},
        "difficulty": "hard",
        "notes": (
            "Power surge aftermath: HW damage (primary) + network config loss (secondary). "
            "Hardware is root cause. P1 due to org-wide impact."
        ),
    },
    {
        "id": "H006",
        "text": (
            "A ransomware note appeared on a file server. All files in the shared drives "
            "show .locked extension. Users cannot open any documents. "
            "The server is still running but shares are inaccessible. "
            "We have not shut it down yet — need guidance."
        ),
        "label": {"category": "security", "priority": "P1", "route": "security_team"},
        "difficulty": "hard",
        "notes": (
            "Active ransomware → security P1. "
            "Operational devastation secondary. Security team must lead IR immediately."
        ),
    },
    {
        "id": "H007",
        "text": (
            "The ERP system (SAP) keeps throwing memory errors and restarting. "
            "This started after a kernel update on the host server. "
            "The DBA says the database connections are fine but the app layer is unstable. "
            "Finance team cannot process payroll which runs today."
        ),
        "label": {"category": "software", "priority": "P1", "route": "dev_team"},
        "difficulty": "hard",
        "notes": (
            "Kernel update broke ERP → software regression. "
            "Payroll processing deadline makes this P1. Dev team + sysadmin collaboration needed."
        ),
    },
    {
        "id": "H008",
        "text": (
            "We are seeing login attempts from a foreign IP range hitting our VPN gateway. "
            "At the same time, our intrusion detection system is alerting on port scans. "
            "One admin account shows a successful auth from this IP range 40 minutes ago. "
            "The account is still active."
        ),
        "label": {"category": "security", "priority": "P1", "route": "security_team"},
        "difficulty": "hard",
        "notes": (
            "Active intrusion with confirmed admin compromise → P1 security incident. "
            "Immediate IR: disable account, block IPs, forensics."
        ),
    },
]

# Convenience accessors
def get_tickets_by_difficulty(difficulty: str):
    return [t for t in TICKETS if t["difficulty"] == difficulty]

def get_all_tickets():
    return TICKETS

def get_ticket_by_id(ticket_id: str):
    for t in TICKETS:
        if t["id"] == ticket_id:
            return t
    return None
