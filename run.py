import papermill as pm
import subprocess
import os

cases = range(1, 58+1)

_ = subprocess.run(['pwsh', 'fix_kernelspec.ps1'])

with open('cde.csv', 'w') as cde_csv:
    cde_csv.write('Case ID,CDE,ratio,drop\n')

for case in cases:
    outname = str(case).zfill(5)
    outfile = os.path.join('./output', f'{outname}.ipynb')
    try:
        pm.execute_notebook('cde.ipynb',
                                    outfile,
                                    parameters=dict(case_id=case))
    except pm.exceptions.PapermillExecutionError as err:
        print(f'Error: {err}')

    _ = subprocess.run(
        ["jupyter", "nbconvert", outfile, "--to=html"]
    )
