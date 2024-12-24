import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2 as pg
import numpy as np
import streamlit as st 

query = """
    WITH program_data AS (
        SELECT 
            program, 
            chain_id, 
            lower(round_id) as round_id, 
            type
        FROM 
            experimental_views.all_rounds_20241029131838 pd
        WHERE 
            round_number IS NOT NULL
            AND lower(round_id) NOT IN (
                '0x911ae126be7d88155aa9254c91a49f4d85b83688',
                '0x40511f88b87b69496a3471cdbe1d3d25ac68e408',
                '0xc08008d47e3deb10b27fc1a75a96d97d11d58cf8',
                '0xb5c0939a9bb0c404b028d402493b86d9998af55e'
            )
    ),
    pl AS (
        SELECT 
            distinct group_id, project_id
            FROM project_lookup),
    donation_summary AS (
        SELECT 
            round_num, 
            chain_id,
            round_id,
            round_name, 
            LOWER(recipient_address) as recipient_address, 
            project_name, 
            project_id, 
            sum(amount_in_usd) as amount_in_usd, 
            count(distinct donor_address) as donors, 
            count(*) as donations 
        FROM all_donations 
        GROUP BY 
            round_num, 
            chain_id,
            round_id,
            round_name, 
            recipient_address, 
            project_name, 
            project_id
    ),
    save as(
    SELECT
        ds.round_num,
        ds.round_name,
        ds.recipient_address,
        ds.project_name,
        ds.project_id,
        ds.amount_in_usd as donation_amount,
        ds.donors,
        ds.donations,
        am.match_amount_in_usd,
        ds.round_id,
        ds.chain_id,
        pl.group_id,
        pd.program,
        pd.type
    FROM donation_summary ds
    LEFT JOIN public.all_matching am ON am.title = ds.project_name AND am.round_id = ds.round_id AND am.chain_id = ds.chain_id
    LEFT JOIN pl ON pl.project_id = ds.project_id
    LEFT JOIN program_data pd ON lower(pd.round_id) = lower(ds.round_id) AND pd.chain_id = ds.chain_id
    WHERE ds.round_num IS NOT NULL)
    SELECT * FROM save 
"""

@st.cache_data(ttl=60*60*24)
def run_query(query, params=None, database="grants"):
    """Run a parameterized query on the specified database and return results as a DataFrame."""
    try:
        conn = pg.connect(host=st.secrets[database]["host"], 
                            port=st.secrets[database]["port"], 
                            dbname=st.secrets[database]["dbname"], 
                            user=st.secrets[database]["user"], 
                            password=st.secrets[database]["password"])
        cur = conn.cursor()
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        col_names = [desc[0] for desc in cur.description]
        results = pd.DataFrame(cur.fetchall(), columns=col_names)
    except pg.Error as e:
        st.warning(f"ERROR: Could not execute the query. {e}")
    finally:
        cur.close()
        conn.close()
    return results

def make_retention_bar_graph(project_data, unique_id):
    # Calculate project status (new, retained, resurrected) for each round
    project_appearances = project_data.groupby([unique_id, 'round_num']).size().reset_index()
    project_appearances = project_appearances.sort_values(['round_num', unique_id])

    # Determine status for each project
    project_status = []
    for _, row in project_data.iterrows():
        prev_rounds = project_appearances[
            (project_appearances[unique_id] == row[unique_id]) & 
            (project_appearances['round_num'] < row['round_num'])
        ]['round_num']
        
        if len(prev_rounds) == 0:
            project_status.append('new')
        elif row['round_num'] - prev_rounds.max() == 1:
            project_status.append('retained')
        else:
            project_status.append('resurrected')

    project_data['status'] = project_status

    # Create summary tables
    projects_by_round = pd.crosstab(project_data['round_num'], project_data['status'])
    projects_by_round_pct = projects_by_round.div(projects_by_round.sum(axis=1), axis=0) * 100

    # Create stacked bar chart
    fig = go.Figure()

    colors = {
        'new': 'rgb(99,110,250)',
        'retained': 'rgb(239,85,59)', 
        'resurrected': 'rgb(0,204,150)'
    }

    for status in ['new', 'retained', 'resurrected']:
        if status in projects_by_round.columns:
            hover_text = [
                f"{status.capitalize()} {count} {pct:.1f}%"
                for round_num, count, pct in zip(
                    projects_by_round.index,
                    projects_by_round[status],
                    projects_by_round_pct[status]
                )
            ]
            
            fig.add_trace(go.Bar(
                x=projects_by_round.index,
                y=projects_by_round[status],
                name=status.capitalize(),
                marker_color=colors[status],
                hovertext=hover_text,
                hoverinfo='text',
                text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(projects_by_round[status], projects_by_round_pct[status])],
                textposition='inside'
            ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Project Participation by Type per Round',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'DarkSlateGrey'}
        },
        xaxis={
            'title': 'Round Number',
            'titlefont': {'size': 16, 'color': 'DarkSlateGrey'},
            'tickfont': {'size': 14, 'color': 'DarkSlateGrey'},
            'showgrid': False,
            'showline': True,
            'linecolor': 'DarkSlateGrey'
        },
        yaxis={
            'title': 'Number of Projects',
            'titlefont': {'size': 16, 'color': 'DarkSlateGrey'},
            'tickfont': {'size': 14, 'color': 'DarkSlateGrey'},
            'showgrid': False,
            'showline': True,
            'linecolor': 'DarkSlateGrey'
        },
        plot_bgcolor='white',
        margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
        barmode='stack',
        hovermode='x unified',
        legend={
            'x': 0.85,
            'y': 0.99,
            'bgcolor': 'rgba(255, 255, 255, 0.5)',
            'bordercolor': 'DarkSlateGrey',
            'borderwidth': 1
        }
    )

    st.plotly_chart(fig)

    # Print summary statistics
    summary_df = pd.DataFrame({
        'Total Projects': projects_by_round.sum(axis=1),
        'New Projects (%)': projects_by_round_pct['new'].round(2),
        'Retained Projects (%)': projects_by_round_pct['retained'].round(2),
        'Resurrected Projects (%)': projects_by_round_pct['resurrected'].round(2),
        'Returning Projects (Retained + Resurrected) (%)': (projects_by_round_pct['retained'] + projects_by_round_pct['resurrected']).round(2)
    })

    st.write("\nSummary Statistics by Round:")
    st.dataframe(summary_df)
    return summary_df

def build_cohort_retention_tables(project_data, unique_id):
    # Determine the range of cohorts and rounds dynamically
    min_cohort = int(project_data['cohort'].min())
    max_cohort = int(project_data['cohort'].max())
    min_round = int(project_data['round_num'].min())
    max_round = int(project_data['round_num'].max())

    # Create a pivot table with cohorts as index and rounds as columns
    cohort_table = pd.pivot_table(project_data, values=unique_id, index='cohort', columns='round_num', aggfunc='nunique', fill_value=0)

    # Ensure cohort_table includes all cohorts and rounds
    all_cohorts = pd.Index(range(min_cohort, max_cohort + 1), name='cohort')
    all_rounds = pd.Index(range(min_round, max_round + 1), name='round_num')
    cohort_table = cohort_table.reindex(index=all_cohorts, columns=all_rounds, fill_value=0)

    # Shift each row to the left until the first non-zero value is in the first column
    for i in range(cohort_table.shape[0]):
        row = cohort_table.iloc[i, :]
        # Only shift if there are any non-zero values
        if row.any():
            # Find the index of the first non-zero value
            non_zero_index = next((index for index, value in enumerate(row) if value != 0), None)
            # Shift the row to the left by the index of the first non-zero value
            if non_zero_index is not None:
                cohort_table.iloc[i, :] = np.roll(row, -non_zero_index)
        else:
            # If row is all zeros, keep it as zeros
            cohort_table.iloc[i, :] = 0

    # Reset the column names to represent the relative round number
    cohort_table.columns = list(range(cohort_table.shape[1]))
    initial_cohort_sizes = cohort_table.iloc[:, 0]
    retention_table = cohort_table.iloc[:, 0:].divide(initial_cohort_sizes.replace(0, 1), axis=0)
    return cohort_table, retention_table

def plot_cohort_retention_heatmap(retention_table):
    # Fill in NaN values for better visualization
    for n in range(len(retention_table)):
        if n != 0:
            retention_table.iloc[n, -n:] = np.nan

    # Create the heatmap using Plotly Express
    fig = px.imshow(retention_table,
                    labels=dict(x="Round Number", y="Cohort", color="Retention Rate"),
                    x=retention_table.columns[:],
                    y=retention_table.index,
                    color_continuous_scale=px.colors.sequential.Blues,
                    text_auto='.2%',
                    aspect="auto")

    # Customize the heatmap layout
    fig.update_layout(
        xaxis=dict(
            title='Rounds Since Cohort Joined',
            side='top'
        ),
        yaxis_title='Cohort',
        plot_bgcolor='white',
        font=dict(size=11),
        height=800
    )

    # Update x-axis and y-axis to show every number
    fig.update_xaxes(tickmode='linear', dtick=1)
    fig.update_yaxes(tickmode='linear', dtick=1)

    # Show the plot
    st.plotly_chart(fig)
### START APP
st.set_page_config(page_title="GG Builder Retention", page_icon="assets/favicon.png", layout="wide")
st.title("📊 Gitcoin Grants Builder Retention")
st.write("This dashboard shows the retention of builders/projects in the Gitcoin Grants program. It shows the number of projects that are new, retained, and resurrected in each round.")

## LOAD DATA
unique_id = 'group_id' # group_id OR project_name OR project_id OR recipient_address
df = run_query(query)
df['round_num'] = pd.to_numeric(df['round_num'], errors='coerce')
df = df.dropna(subset=['round_num'])
df['cohort'] = df.groupby(unique_id)['round_num'].transform('min')

## MAKE BAR GRAPH
st.subheader("Builders by Round")
st.write("This bar graph shows the number of projects that are new, retained, and resurrected in each round.")
st.write("Resurrected projects are projects that have been active in previous rounds but were not active in the immediate previous round. Retained projects are projects that have been active in the previous round and are active in the current round.")
st.write("Returning projects is retained + resurrected.")
summary_df = make_retention_bar_graph(df, unique_id)

## MAKE HEATMAP
cohort_table, retention_table = build_cohort_retention_tables(df, unique_id)
st.subheader("Retention Heatmap")
st.write("The heatmap illustrates the retention rates of projects across different cohorts and rounds, providing a visual representation of project longevity.")
st.write("The heatmap is interactive. You can hover over each cell to see the retention rate for that cohort and round.")
st.write("A cohort is a group of projects that joined the program in the same round. The retention rate is the percentage of projects from a cohort that are still active in a given round.")
plot_cohort_retention_heatmap(retention_table)

## DRILL DOWN INTO A ROUND
st.subheader("Drill Down into a Round")
round_num = st.selectbox("Select a program number to drill down into", df['round_num'].unique())
round_data = df[df['round_num'] == round_num]
round_name = st.selectbox("Select a round to drill down into", ['All'] + round_data['round_name'].unique().tolist())
if round_name != 'All':
    round_data = round_data[(round_data['round_name'] == round_name)]
round_data = round_data[['status', 'project_name'] + [col for col in round_data.columns if col not in ['status', 'project_name']]]
st.write(round_data)
