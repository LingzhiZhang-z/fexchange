function shq(s,    out, i, c) {
    out = "'"
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "'") {
            out = out "'\\''"
        } else {
            out = out c
        }
    }
    return out "'"
}

function trim(s) {
    sub(/^[ \t\r\n]+/, "", s)
    sub(/[ \t\r\n]+$/, "", s)
    return s
}

BEGIN {
    in_rs = 0
    rs_idx = 0
}

{
    line = trim($0)
    if (line == "") {
        next
    }
    if (line == "<RS") {
        in_rs = 1
        next
    }
    if (line == "RS>") {
        in_rs = 0
        next
    }
    if (in_rs) {
        rs[rs_idx++] = line
        next
    }
    key = $1
    $1 = ""
    value = trim($0)
    data[key] = value
}

END {
    split(data["ATOMS"], atoms, /[ \t]+/)
    material = FILENAME
    sub(/^.*\/data\/test\//, "", material)
    sub(/\/input\/.*$/, "", material)
    prefix = data["PREFIX"]
    family = "all"
    if (prefix ~ /_1st_/) {
        family = "1st"
    } else if (prefix ~ /_2nd_/) {
        family = "2nd"
    }

    print "PREFIX=" shq(prefix)
    print "MATERIAL=" shq(material)
    print "FAMILY=" shq(family)
    print "FILE_HR=" shq(data["FILE_HR"])
    print "SLATER_RATIOS=" shq(data["SLATER_RATIOS"])
    print "U_LIST=" shq(data["Us"])
    print "RATIO_LIST=" shq(data["ratios"])
    print "SPINOR=" shq(tolower(data["SPINOR"]))
    print "NDIMS=" shq(data["NDIMS"])
    print "ATOM_I=" shq(atoms[1] - 1)
    print "ATOM_J=" shq(atoms[2] - 1)
    print "LIGAND_1=" shq(atoms[3] - 1)
    print "LIGAND_2=" shq(atoms[4] - 1)
    print "CELL_I=" shq(rs[0])
    print "CELL_J=" shq(rs[1])
    print "CELL_L1=" shq(rs[2])
    print "CELL_L2=" shq(rs[3])
    print "REFERENCE_DIR=" shq(repo_root "/data/test/" material "/" prefix)
}
