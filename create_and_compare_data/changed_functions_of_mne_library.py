''' 
To be able to get the topomap function to work with the provided .elc file I had to make some minor adjustements to some
functions of the mne library. This is an ugly solution but it worked for me. If the library is installed, these functions 
will be different. To run the file topomap_eeg.py these minor adjustments are necessary inside the _read_elc function in the 
_standard_montage_utils.py file (or changing the .elc file itself).

'''

def _read_elc(fname, head_size):
    """Read .elc files.

    Parameters
    ----------
    fname : str
        File extension is expected to be '.elc'.
    head_size : float | None
        The size of the head in [m]. If none, returns the values read from the
        file with no modification.

    Returns
    -------
    montage : instance of DigMontage
        The montage in [m].
    """
    fid_names = ("Nz", "LPA", "RPA")

    ch_names_, pos = [], []
    with open(fname) as fid:
        # _read_elc does require to detect the units. (see _mgh_or_standard)
        for line in fid:
            if "UnitPosition" in line:
                units = line.split()[1]
                scale = dict(m=1.0, mm=1e-3)[units]
                break
        else:
            raise RuntimeError("Could not detect units in file %s" % fname)
        for line in fid:
            if "Positions\n" in line:
                break
        pos = []
        for line in fid:
            if "Labels\n" in line:
                break
            pos.append(list(map(float, line.split()[-3:])))
        for line in fid:
            if not line or not set(line) - {" "}:
                break
            #ch_names_.append(line.strip(" ").strip("\n"))
            ch_names_.append(line.split())
    ch_names_ = ch_names_[0]
    pos = np.array(pos) * scale
    if head_size is not None:
        pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

    ch_pos = _check_dupes_odict(ch_names_, pos)
    nasion, lpa, rpa = (ch_pos.pop(n, None) for n in fid_names)

    return make_dig_montage(
        ch_pos=ch_pos, coord_frame="unknown", nasion=nasion, lpa=lpa, rpa=rpa
    )