"""Interactive filter panel for the profile table.

``render(st, df, scope=...)`` draws the filters, applies them LIVE (no Apply button, so results can never be
stale), shows the matching profiles and offers CSV / formatted-Excel downloads of the chosen columns.
All matching logic lives in ``profile_filter``; this module is only widgets and state.

Streamlit is passed in (``st``) rather than imported so the panel can be tested with a fake that enforces
Streamlit's rules for keys and session state.

Rules this panel follows (the original filter broke each of them):

* every widget key contains a STABLE filter id, never the filter's position in the list, so removing a filter
  cannot hand its values (or its uploaded file) to the next one;
* "Select all" / "Clear" change a widget's value through ``on_click`` callbacks, the only way that works once
  the widget exists (a ``default=`` is ignored after the first run);
* widget state that no longer fits the data (a column that is gone) is dropped before the widget is drawn;
* filters that cannot be applied are listed on screen, never skipped silently.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

import profile_filter as pf

__all__ = ["LOGIC_LABELS", "key", "panel_key", "render"]

LOGIC_LABELS = {
    "contains": "contains",
    "not_contains": "does not contain",
    "in_list": "is one of a list",
    "not_in_list": "is not in a list",
    "is_empty": "is empty",
    "not_empty": "is not empty",
    "num_ge": "number is at least (>=)",
    "num_le": "number is at most (<=)",
    "num_eq": "number equals (=)",
    "num_between": "number is between",
}
MATCH_LABELS = {"item": "whole item (exact)", "word": "whole word", "substring": "text anywhere"}
MATCH_HELP = (
    "whole item: one entry of a ; list, so 'java' does NOT match 'JavaScript' and 'react' does NOT match "
    "'React Native'.  whole word: the word anywhere in the text.  text anywhere: plain substring, so 'java' also "
    "matches 'JavaScript'."
)
COMBINE_CONTAINS = {"OR": "any of these", "AND": "all of these"}
COMBINE_NOT = {"AND": "none of these", "OR": "not all of these"}
FREE_TEXT_COLUMNS = frozenset({"phone", "resume_link", "github_profile"})   # default match: text anywhere


def is_tech_column(name: str) -> bool:
    return name.startswith("tech_") or name in {"github_tech_stack", "unrecognized_technologies"}


def key(scope: str, fid: int, name: str) -> str:
    """Widget key of one control of one filter."""
    return f"flt_{scope}_{fid}_{name}"


def panel_key(scope: str, name: str) -> str:
    """Session-state key of a panel-level control or value (ids, columns, mode, add / clear buttons...)."""
    return f"flt_{scope}__{name}"


_pkey = panel_key


# --------------------------------------------------------------------------- #
# state + callbacks (callbacks run before the next script run)
# --------------------------------------------------------------------------- #

def _ids(st, scope: str) -> list[int]:
    ik, nk = _pkey(scope, "ids"), _pkey(scope, "next")
    if ik not in st.session_state:
        st.session_state[ik] = [0]
        st.session_state[nk] = 1
    return st.session_state[ik]


def _add(st, scope: str) -> None:
    nk = _pkey(scope, "next")
    st.session_state[_pkey(scope, "ids")].append(st.session_state[nk])
    st.session_state[nk] += 1


def _remove(st, scope: str, fid: int) -> None:
    ids = st.session_state[_pkey(scope, "ids")]
    if fid in ids:
        ids.remove(fid)
    if not ids:                                   # never leave the panel without a filter row
        _add(st, scope)


def _clear(st, scope: str) -> None:
    nk = _pkey(scope, "next")
    st.session_state[_pkey(scope, "ids")] = [st.session_state[nk]]   # fresh id => fresh, empty widgets
    st.session_state[nk] += 1


def _set_columns(st, state_key: str, columns: list[str]) -> None:
    st.session_state[state_key] = list(columns)


# --------------------------------------------------------------------------- #
# one filter
# --------------------------------------------------------------------------- #

def _render_filter(st, scope: str, fid: int, n: int, columns: list[str], resolver) -> pf.FilterSpec:
    def k(name: str) -> str:
        return key(scope, fid, name)

    stored = st.session_state.get(k("col"))
    if stored is not None and stored not in columns:      # data changed: drop what no longer fits
        del st.session_state[k("col")]

    with st.container(border=True):
        head, remove = st.columns([8, 1])
        head.markdown(f"**Filter {n}**")
        remove.button("Remove", key=k("remove"), on_click=_remove, args=(st, scope, fid))

        left, right = st.columns(2)
        column = left.selectbox("Column", columns, key=k("col"))
        logic = right.selectbox("Condition", list(pf.LOGICS), format_func=LOGIC_LABELS.__getitem__, key=k("logic"))
        spec = pf.FilterSpec(column=column, logic=logic)

        if logic in pf.TEXT_LOGICS:
            text = st.text_input(
                "Keywords (comma separated)", key=k("kw"), placeholder="python, react",
                help="Several keywords: separate them with commas.",
            )
            values = pf.parse_keywords(text)
            a, b = st.columns(2)
            free_text = column in FREE_TEXT_COLUMNS      # phone / links: "text anywhere"; lists: "whole item"
            spec.match = a.selectbox(
                "Match", list(pf.MATCH_TYPES), index=2 if free_text else 0,
                format_func=MATCH_LABELS.__getitem__, key=k("match_free" if free_text else "match_list"), help=MATCH_HELP,
            )
            if logic == "contains":
                spec.combine = b.selectbox("Keywords combine as", list(COMBINE_CONTAINS), format_func=COMBINE_CONTAINS.__getitem__, key=k("comb_contains"))
            else:
                spec.combine = b.selectbox("Keywords combine as", list(COMBINE_NOT), format_func=COMBINE_NOT.__getitem__, key=k("comb_not"))
            if resolver is not None and is_tech_column(column) and values:
                if st.checkbox("Understand technology aliases (nodejs = Node.js)", value=True, key=k("alias")):
                    resolved = [resolver(v) for v in values]
                    if resolved != values:
                        st.caption("Searching for: " + ", ".join(resolved))
                    values = resolved
            spec.values = values

        elif logic in pf.LIST_LOGICS:
            source = st.radio("Values from", ["Type or paste", "Upload a file"], horizontal=True, key=k("src"))
            if source == "Type or paste":
                text = st.text_area(
                    "Values (one per line, or separated by commas)", key=k("vals"), height=100,
                    help="With several lines each line is one value, so a value may contain commas.",
                )
                spec.values = pf.parse_values(text)
            else:
                spec.values = _values_from_file(st, k)
            spec.phone_mode = st.checkbox(
                "Compare as phone numbers (ignore spaces, +91 and a leading 0)",
                value="phone" in column.lower(), key=k("phone"),
            )

        elif logic in pf.NUMBER_LOGICS:
            if logic == "num_between":
                a, b = st.columns(2)
                spec.low = a.number_input("Minimum", value=0.0, step=1.0, key=k("low"))
                spec.high = b.number_input("Maximum", value=10.0, step=1.0, key=k("high"))
            else:
                number = st.number_input("Number", value=1.0, step=1.0, key=k("num"))
                if logic == "num_le":
                    spec.high = number
                else:
                    spec.low = number
    return spec


def _values_from_file(st, k: Callable[[str], str]) -> list[str]:
    upload = st.file_uploader("Excel (.xlsx), CSV or text file with the values", type=["xlsx", "csv", "txt"], key=k("file"))
    if upload is None:
        return []
    header = st.checkbox("The first row is a header (not a value)", value=True, key=k("hdr"))
    data = upload.getvalue()
    probe = pf.read_value_list(data, upload.name, None, header)
    if probe.error and not probe.columns:
        st.error(probe.error)
        return []
    chosen = probe.column
    if len(probe.columns) > 1:
        # the widget key carries the file identity, so a new file never inherits a stale column choice
        chosen = st.selectbox("Column with the values", probe.columns, key=k(f"xcol_{upload.name}_{len(data)}_{header}"))
    load = pf.read_value_list(data, upload.name, chosen, header)
    if load.error:
        st.warning(load.error)
        return []
    preview = ", ".join(load.values[:8]) + (" ..." if len(load.values) > 8 else "")
    st.caption(f"{len(load.values)} unique value(s) loaded: {preview}")
    return load.values


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #

def render(
    st,
    df: pd.DataFrame,
    scope: str = "results",
    *,
    alias_resolver: Callable[[str], str] | None = None,
    csv_builder: Callable[[pd.DataFrame, list[str]], bytes] | None = None,
    xlsx_builder: Callable[[pd.DataFrame, list[str]], bytes] | None = None,
    file_stem: str = "filtered_profiles",
) -> pf.FilterResult:
    """Draw the filter panel for ``df`` (all columns must hold text) and return the ``FilterResult``.

    ``scope`` keeps several panels on one page apart. ``alias_resolver`` (see ``profile_filter.alias_resolver``)
    lets technology columns understand aliases. ``csv_builder`` / ``xlsx_builder`` produce the download bytes for
    (frame, chosen columns); defaults write UTF-8-with-BOM CSV and the formatted workbook of ``xlsx_export``.
    """
    columns = list(df.columns)
    if not columns:
        st.warning("The table has no columns, so there is nothing to filter.")
        return pf.apply_filters(df, [])
    ids = list(_ids(st, scope))

    specs = [_render_filter(st, scope, fid, n, columns, alias_resolver) for n, fid in enumerate(ids, start=1)]

    c1, c2, c3 = st.columns([1, 1, 3])
    c1.button("Add filter", key=_pkey(scope, "add"), on_click=_add, args=(st, scope))
    c2.button("Clear filters", key=_pkey(scope, "clear"), on_click=_clear, args=(st, scope))
    mode = c3.radio("Show profiles that match", ["ALL of the filters", "ANY of the filters"], horizontal=True, key=_pkey(scope, "mode"))
    result = pf.apply_filters(df, specs, combine="OR" if mode.startswith("ANY") else "AND")

    if result.active > 0:
        for message in result.ignored:               # some filters work, others do not: say so loudly
            st.warning(f"Not applied: {message}")
    for message in result.notes:
        st.caption(message)
    if result.active == 0:
        reasons = "; ".join(result.ignored)
        st.info(f"No filter is complete yet, so all {result.total} profiles are shown." + (f" ({reasons})" if reasons else ""))
    elif result.matched == 0:
        st.warning(f"No profile matches (0 of {result.total}). Open the list below to see how many each filter matches on its own.")
    else:
        st.success(f"{result.matched} of {result.total} profiles match.")
    if result.per_filter:
        with st.expander("How many profiles each filter matches on its own"):
            for description, count in result.per_filter:
                st.write(f"{count} - {description}")

    # ---- columns to show / download
    ck = _pkey(scope, "cols")
    if ck not in st.session_state:
        st.session_state[ck] = list(columns)
    else:
        valid = [c for c in st.session_state[ck] if c in columns]
        if valid != st.session_state[ck]:
            st.session_state[ck] = valid
    s1, s2 = st.columns(2)
    s1.button("Select all columns", key=_pkey(scope, "all"), on_click=_set_columns, args=(st, ck, columns))
    s2.button("Clear columns", key=_pkey(scope, "none"), on_click=_set_columns, args=(st, ck, []))
    selected = st.multiselect("Columns to show and download", columns, key=ck)

    if not selected:
        st.warning("Choose at least one column to show and download.")
        return result

    st.dataframe(result.frame[selected], hide_index=True)

    csv_bytes = (csv_builder or _default_csv)(result.frame, selected)
    st.download_button(
        f"Download filtered CSV ({result.matched} profiles)", data=csv_bytes, file_name=f"{file_stem}.csv",
        mime="text/csv", key=_pkey(scope, "dl_csv"),
    )
    try:
        xlsx_bytes = (xlsx_builder or _default_xlsx)(result.frame, selected)
    except ImportError:
        st.info("For a formatted Excel download install xlsxwriter:  pip install xlsxwriter")
    else:
        st.download_button(
            f"Download filtered Excel ({result.matched} profiles)", data=xlsx_bytes, file_name=f"{file_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=_pkey(scope, "dl_xlsx"),
        )
    return result


def _default_csv(frame: pd.DataFrame, columns: list[str]) -> bytes:
    return frame[columns].to_csv(index=False).encode("utf-8-sig")


def _default_xlsx(frame: pd.DataFrame, columns: list[str]) -> bytes:
    import xlsx_export
    return xlsx_export.dataframe_to_xlsx(frame, columns)
