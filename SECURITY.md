# Security policy

Report a vulnerability privately through GitHub's "Report a
vulnerability" button on this repository's Security tab. Please do not
open a public issue for it. You will get a reply within 7 days.

Scope: the code in this repository. The benchmark runs offline on one
private VM against a public dataset (UCI default of credit card
clients) and serves nothing. No credential is stored in this
repository; the Jev claim check calls an external helper that holds
its key outside this repository (scripts/jev_claim_check.py).
