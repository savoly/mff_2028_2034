Get-ChildItem -Recurse -Filter *.ipynb |
    Where-Object {
        $_.FullName -notmatch '\\.virtual_documents\\' -and
        $_.FullName -notmatch '\\.ipynb_checkpoints\\'
    } |
    ForEach-Object -Parallel {
      # clear all outputs in-place
      jupyter nbconvert --to notebook --inplace --clear-output $PSItem.FullName 2>$null

      # load notebook JSON
      $nb = Get-Content -LiteralPath $PSItem.FullName -Raw | ConvertFrom-Json

      # drop empty cells
      $nb.cells = $nb.cells | ? { $_.source -and (($_.source -join '') -match '\S') }

      # write cleaned notebook back
      $nb | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $PSItem.FullName -Encoding UTF8
    }
