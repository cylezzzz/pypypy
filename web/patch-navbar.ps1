# X:\pypygennew\web\patch-navbar.ps1
$root  = "X:\pypygennew\web"

# nur Top-Level HTML (nicht rekursiv)
$files = Get-ChildItem -Path $root -File -Filter *.html

foreach ($f in $files) {
    $c = Get-Content -Raw -LiteralPath $f.FullName

    # 1) <head> → ui.css injizieren (nur wenn nicht vorhanden)
    if ($c -notmatch 'assets/ui.css') {
        # WICHTIG: `$1` in doppelten Anführungszeichen mit Backtick escapen
        $c = $c -replace '(<head[^>]*>)', "`$1`r`n  <link rel=`"stylesheet`" href=`"assets/ui.css`">"
    }

    # 2) <body> → Navbar-Placeholder injizieren (nur wenn nicht vorhanden)
    if ($c -notmatch 'data-include="navbar"') {
        $c = $c -replace '(<body[^>]*>)', "`$1`r`n  <div data-include=`"navbar`"></div>"
    }

    # 3) vor </body> → Scripts injizieren (nur wenn nicht vorhanden)
    $needInclude = $c -notmatch 'assets/include.js'
    $needOverlay = $c -notmatch 'assets/overlay-player.js'

    if ($needInclude -or $needOverlay) {
        $footer = ""
        if ($needInclude) { $footer += "  <script src=`"assets/include.js`"></script>`r`n" }
        if ($needOverlay) { $footer += "  <script src=`"assets/overlay-player.js`"></script>`r`n" }

        # KEINE + Verkettung hinter -replace ohne Klammern!
        # `$1` wieder escapen, damit die Regex-Ersetzung die Capture-Group bekommt.
        $c = $c -replace '(</body>)', "$footer`$1"
        # Alternativ ginge auch:
        # $c = $c -replace '(</body>)', ($footer + '$1')
    }

    Set-Content -LiteralPath $f.FullName -Value $c -Encoding UTF8
    "Patched $($f.Name)"
}

"Done."
