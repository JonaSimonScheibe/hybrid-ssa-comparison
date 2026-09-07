# Restliche Schritte (brauchen deinen GitHub-Login)

Das lokale Repository ist fertig angelegt und committet. Es fehlt nur noch die Verbindung
zu GitHub — drei Schritte, ca. 3 Minuten. Diese Datei danach löschen:
`git rm SETUP.md && git commit -m "Remove setup notes" && git push`

## 1. SSH-Schlüssel bei GitHub hinterlegen

Schlüssel anzeigen und den ganzen Text kopieren (nur die `.pub`-Datei — die Datei ohne
`.pub` ist der private Schlüssel und wird niemals weitergegeben):

```bash
cat ~/.ssh/id_ed25519.pub
```

github.com → Profilbild oben rechts → **Settings** → **SSH and GPG keys** →
**New SSH key** → Title z. B. „Tuxedo Laptop", Key type „Authentication key", Text einfügen
→ **Add SSH key**.

Prüfen:

```bash
ssh -T git@github.com
```

Erwartet: `Hi <username>! You've successfully authenticated, but GitHub does not provide
shell access.` Die zweite Hälfte ist normal und kein Fehler.

## 2. Leeres Repository auf GitHub anlegen

github.com → **+** oben rechts → **New repository**

- Repository name: `hybrid-ssa-comparison`
- Description: `Numerical comparison of hybrid stochastic simulation algorithms for reaction kinetics`
- **Public**
- Unter „Initialize this repository with" **nichts ankreuzen** — kein README, keine
  .gitignore, keine Lizenz. Diese Dateien liegen bereits hier. Legt GitHub eigene an, haben
  beide Seiten unterschiedliche Historien und der erste Push wird mit
  `rejected … fetch first` abgewiesen.

→ **Create repository**

## 3. Verbinden und pushen

Im Terminal in diesem Ordner (`<dein-username>` ersetzen):

```bash
cd ~/Schreibtisch/Studium/Semester4/hybrid-ssa-comparison
git remote add origin git@github.com:<dein-username>/hybrid-ssa-comparison.git
git push -u origin main
```

`origin` ist der übliche Name für „das entfernte Repository", `-u` merkt sich die Zuordnung,
sodass künftig ein blankes `git push` genügt.

## Danach

Der Ablauf ist ab jetzt immer derselbe:

```bash
git add -A
git commit -m "Beschreibung der Änderung"
git push
```

Vor dem Hineinkopieren von Dateien lohnt ein Blick auf `.gitignore`: `*.csv`, `*.log`,
`__pycache__/` und der Inhalt von `results/` sind bewusst ausgeschlossen, damit die
Ergebnisdaten aus dem Projektarbeit-Ordner nicht versehentlich im Repository landen.
Ob eine Datei ignoriert wird, sagt `git check-ignore -v <datei>`.
