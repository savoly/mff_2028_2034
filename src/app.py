import pandas as pd
import streamlit as st

from mff.new_cap import compute_dabis_support_summary, read_base_data, find_flat_rate
from mff.utils import c_round

st.set_page_config(page_title="MFF számítások", layout="wide")


@st.cache_data
def load_data(year: int) -> pd.DataFrame:
    return read_base_data(year)


def format_with_space(n: float) -> str:
    return f"{n:,.2f}".replace(",", " ").replace(".", ",")


def calculate_subs(data, budget, yfs_per_ha, redist_per_ha1, redist_per_ha2):
    solution_with_yfs = find_flat_rate(
        data, budget, yfs_per_ha, (redist_per_ha1, redist_per_ha2)
    )

    return pd.DataFrame(
        {
            "YFS per/ha": [yfs_per_ha],
            "DABIS per/ha": [c_round(solution_with_yfs.root, 2)],
            "REDIST per/ha": [(redist_per_ha1, redist_per_ha2)],
        }
    )


def main():
    data = load_data(2024)

    st.title("MFF számítások")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input(
            "Költségvetés (EUR)",
            value=943_167_086.78,
            min_value=0.0,
            step=1_000_000.0,
            format="%.2f",
        )
        # st.caption(f"Jelenlegi költségvetés: **{format_with_space(budget)}** EUR")

        yfs_per_ha = st.number_input(
            "Fiatal gazdálkodók támogatása (EUR/ha)",
            value=90.0,
            min_value=0.0,
            step=5.0,
        )

    with col2:
        redist_step1 = st.number_input(
            "Redisztribúciós lépcső 1–10 ha (EUR/ha)",
            value=80.0,
            min_value=0.0,
            step=5.0,
        )
        redist_step2 = st.number_input(
            "Redisztribúciós lépcső 10–150 ha (EUR/ha)",
            value=40.0,
            min_value=0.0,
            step=5.0,
        )

    st.markdown("---")

    if st.button("Számítás indítása"):
        try:
            summary = compute_dabis_support_summary(data)
            subs_df = calculate_subs(
                data,
                budget,
                yfs_per_ha,
                redist_step1,
                redist_step2,
            )

            st.subheader("DABIS támogatás összefoglaló")
            st.dataframe(summary, use_container_width=True)

            st.subheader("Paraméterekből számolt kulcsértékek")
            st.dataframe(subs_df, use_container_width=True)

        except Exception as e:
            st.error(f"Hiba a számítás során: {e}")


if __name__ == "__main__":
    main()
