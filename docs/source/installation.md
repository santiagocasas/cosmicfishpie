Installation
============

**cosmicfishpie** supports Python >= 3.8.

## Installing with `pip`

**cosmicfishpie** is available [on PyPI](https://pypi.org/project/cosmicfishpie/). Just run

```bash
pip install cosmicfishpie
```

## Installing from source

To install **cosmicfishpie** from source, first clone [the repository](https://github.com/santiagocasas/cosmicfishpie):

```bash
git clone https://github.com/santiagocasas/cosmicfishpie.git
cd cosmicfishpie
```

Then run

```bash
pip install -e .
```

## Installing additional external data

If **cosmicfishpie** was installed using `pip` additional data files that are used for the different experiments are not downloaded by default. To download these we have provided a small script inside of the configs folder. It will download these data files directly from github repository.

For this just run

```bash
python install_external_data.py
```

and follow the command prompts in the terminal.

By default it will try to download the files to the configs folder but the user can specify another path
