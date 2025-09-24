import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
import io
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SeaQuence: AI Biodiversity Explorer",
    page_icon="🌊",
    layout="wide"
)

class SeaQuenceAnalyzer:
    """Main class for eDNA biodiversity analysis"""
    
    def __init__(self):
        self.sequences = []
        self.sequence_types = []
        self.taxa_labels = []
        self.kmers = []
        self.embeddings = None
        self.clusters = None
        self.classifier = None
        
    def detect_sequence_type(self, sequence):
        """Detect if sequence is DNA/RNA or Protein"""
        sequence = str(sequence).upper()
        dna_chars = set('ATCGUN')  # Including N for ambiguous bases, U for RNA
        protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
        
        seq_chars = set(sequence)
        
        # If more than 95% are DNA characters, classify as DNA
        if len(seq_chars.intersection(dna_chars)) / len(seq_chars) > 0.95:
            return 'DNA'
        # If more than 90% are protein characters, classify as Protein
        elif len(seq_chars.intersection(protein_chars)) / len(seq_chars) > 0.90:
            return 'Protein'
        else:
            return 'Unknown'
    
    def extract_taxa_from_header(self, header):
        """Extract taxonomic information from FASTA header"""
        # Look for patterns like [Species name] or |Species name|
        patterns = [
            r'\[([A-Z][a-z]+ [a-z]+)\]',  # [Homo sapiens]
            r'\|([A-Z][a-z]+ [a-z]+)\|',  # |Homo sapiens|
            r'([A-Z][a-z]+ [a-z]+)',      # Direct species name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, header)
            if match:
                return match.group(1)
        
        # If no species found, try to extract genus
        genus_match = re.search(r'([A-Z][a-z]+)', header)
        if genus_match:
            return genus_match.group(1) + ' sp.'
            
        return 'Unknown'
    
    def generate_kmers(self, sequence, k):
        """Generate k-mers from sequence"""
        sequence = str(sequence).upper()
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmers.append(sequence[i:i+k])
        return kmers
    
    def parse_fasta(self, fasta_content):
        """Parse FASTA file and extract sequences with metadata"""
        fasta_io = io.StringIO(fasta_content)
        
        sequences = []
        headers = []
        
        for record in SeqIO.parse(fasta_io, "fasta"):
            sequences.append(str(record.seq))
            headers.append(record.description)
        
        # Process each sequence
        for i, (seq, header) in enumerate(zip(sequences, headers)):
            seq_type = self.detect_sequence_type(seq)
            taxa = self.extract_taxa_from_header(header)
            
            # Generate k-mers based on sequence type
            if seq_type == 'DNA':
                seq_kmers = self.generate_kmers(seq, k=6)
            elif seq_type == 'Protein':
                seq_kmers = self.generate_kmers(seq, k=3)
            else:
                seq_kmers = self.generate_kmers(seq, k=6)  # Default to DNA
                
            self.sequences.append(seq)
            self.sequence_types.append(seq_type)
            self.taxa_labels.append(taxa)
            self.kmers.append(' '.join(seq_kmers))  # Join kmers as text for vectorization
    
    def create_embeddings(self, method='bow'):
        """Create embeddings from k-mers"""
        if method == 'bow':
            # Bag-of-k-mers using CountVectorizer
            vectorizer = CountVectorizer(max_features=1000, token_pattern=r'\b\w+\b')
            self.embeddings = vectorizer.fit_transform(self.kmers).toarray()
            self.vectorizer = vectorizer
        
        return self.embeddings
    
    def perform_clustering(self, method='kmeans', n_clusters=5):
        """Perform unsupervised clustering"""
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Run create_embeddings first.")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = clusterer.fit_predict(self.embeddings)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            self.clusters = clusterer.fit_predict(self.embeddings)
        
        return self.clusters
    
    def train_classifier(self):
        """Train supervised classifier on known taxa"""
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Run create_embeddings first.")
        
        # Filter out unknown taxa for training
        known_indices = [i for i, taxa in enumerate(self.taxa_labels) if taxa != 'Unknown']
        
        if len(known_indices) < 2:
            return None, "Not enough labeled data for classification"
        
        X = self.embeddings[known_indices]
        y = [self.taxa_labels[i] for i in known_indices]
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store label encoder for predictions
        self.label_encoder = le
        
        return accuracy, le.inverse_transform(y_test), le.inverse_transform(y_pred)
    
    def calculate_abundance(self):
        """Calculate abundance per cluster/taxa"""
        if self.clusters is None:
            return None
        
        # Count sequences per cluster
        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        
        # Count sequences per taxa
        taxa_counts = pd.Series(self.taxa_labels).value_counts()
        
        return cluster_counts, taxa_counts
    
    def visualize_clusters(self, method='pca'):
        """Create 2D visualization of clusters"""
        if self.embeddings is None or self.clusters is None:
            return None
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(self.embeddings)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
            coords_2d = reducer.fit_transform(self.embeddings)
        
        return coords_2d

def create_sample_fasta():
    """Create sample FASTA data for demo"""
    sample_data = """>seq1 [Homo sapiens] human sequence
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
>seq2 [Escherichia coli] bacterial sequence  
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
>seq3 [Saccharomyces cerevisiae] yeast sequence
TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>seq4 [Drosophila melanogaster] fly sequence
CGAATTCGAATTCGAATTCGAATTCGAATTCGAATTCGAATTCGAATTCGAATTCGAATT
>seq5 Unknown organism
TTGCAATTGCAATTGCAATTGCAATTGCAATTGCAATTGCAATTGCAATTGCAATTGCAA
>seq6 [Caenorhabditis elegans] nematode sequence
AGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTC
>seq7 Novel sequence cluster A
GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC
>seq8 [Arabidopsis thaliana] plant sequence
CTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGA"""
    
    return sample_data

# Streamlit App
def main():
    st.title("🌊 SeaQuence: AI Biodiversity Explorer")
    st.markdown("*Discover known and novel taxa from eDNA sequences using AI*")
    
    # Sidebar
    st.sidebar.header("📊 Analysis Parameters")
    clustering_method = st.sidebar.selectbox("Clustering Method", ["kmeans", "dbscan"])
    n_clusters = st.sidebar.slider("Number of Clusters (K-means)", 2, 15, 5)
    viz_method = st.sidebar.selectbox("Visualization Method", ["pca", "tsne"])
    
    # Main interface
    st.header("📁 Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload FASTA file",
            type=['fasta', 'fa', 'fas', 'txt'],
            help="Upload your eDNA sequences in FASTA format"
        )
    
    with col2:
        use_sample = st.button("🎯 Use Sample Data", help="Load demo data for testing")
    
    # Initialize analyzer
    analyzer = SeaQuenceAnalyzer()
    
    # Process data
    if uploaded_file is not None or use_sample:
        if use_sample:
            fasta_content = create_sample_fasta()
            st.success("Sample data loaded!")
        else:
            fasta_content = uploaded_file.getvalue().decode('utf-8')
            st.success(f"File uploaded: {uploaded_file.name}")
        
        with st.spinner("🧬 Parsing sequences..."):
            analyzer.parse_fasta(fasta_content)
        
        # Display basic stats
        st.header("📈 Sequence Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sequences", len(analyzer.sequences))
        with col2:
            dna_count = analyzer.sequence_types.count('DNA')
            st.metric("DNA Sequences", dna_count)
        with col3:
            protein_count = analyzer.sequence_types.count('Protein')
            st.metric("Protein Sequences", protein_count)
        with col4:
            known_taxa = len([x for x in analyzer.taxa_labels if x != 'Unknown'])
            st.metric("Known Taxa", known_taxa)
        
        # Show sequence type distribution
        st.subheader("Sequence Type Distribution")
        type_counts = pd.Series(analyzer.sequence_types).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        type_counts.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72', '#F18F01'])
        plt.title("Distribution of Sequence Types")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Create embeddings and perform analysis
        with st.spinner("🔬 Creating embeddings and clustering..."):
            analyzer.create_embeddings(method='bow')
            analyzer.perform_clustering(method=clustering_method, n_clusters=n_clusters)
            coords_2d = analyzer.visualize_clusters(method=viz_method)
        
        # Clustering Results
        st.header("🎯 Clustering Analysis")
        
        if coords_2d is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot with cluster colors
            scatter = ax.scatter(
                coords_2d[:, 0], coords_2d[:, 1], 
                c=analyzer.clusters, 
                cmap='tab10', 
                alpha=0.7,
                s=50
            )
            
            # Add taxa labels for known sequences
            for i, (x, y, taxa) in enumerate(zip(coords_2d[:, 0], coords_2d[:, 1], analyzer.taxa_labels)):
                if taxa != 'Unknown' and i % 3 == 0:  # Show every 3rd label to avoid crowding
                    ax.annotate(taxa.split()[0], (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, alpha=0.7)
            
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'Sequence Clusters ({viz_method.upper()} visualization)')
            plt.xlabel(f'{viz_method.upper()} Component 1')
            plt.ylabel(f'{viz_method.upper()} Component 2')
            st.pyplot(fig)
            
            # Cluster summary
            cluster_df = pd.DataFrame({
                'Sequence_ID': range(len(analyzer.sequences)),
                'Taxa': analyzer.taxa_labels,
                'Cluster': analyzer.clusters,
                'Type': analyzer.sequence_types
            })
            
            st.subheader("Cluster Summary")
            cluster_summary = cluster_df.groupby('Cluster').agg({
                'Taxa': lambda x: list(x.unique()),
                'Type': lambda x: list(x.unique()),
                'Sequence_ID': 'count'
            }).rename(columns={'Sequence_ID': 'Count'})
            
            st.dataframe(cluster_summary)
        
        # Classification
        st.header("🎓 Taxonomic Classification")
        
        with st.spinner("🤖 Training classifier..."):
            accuracy, y_true, y_pred = analyzer.train_classifier()
        
        if accuracy is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Classification Accuracy", f"{accuracy:.2%}")
            
            with col2:
                novel_clusters = len(set(analyzer.clusters[analyzer.taxa_labels.index('Unknown'):]))
                st.metric("Potential Novel Clusters", novel_clusters)
            
            # Show classification examples
            st.subheader("Classification Examples")
            if len(y_true) > 0:
                examples_df = pd.DataFrame({
                    'True Taxa': y_true[:10],  # Show first 10 examples
                    'Predicted Taxa': y_pred[:10]
                })
                st.dataframe(examples_df)
        else:
            st.warning("Not enough labeled data for classification training")
        
        # Abundance Analysis
        st.header("📊 Abundance Analysis")
        
        cluster_counts, taxa_counts = analyzer.calculate_abundance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Abundance")
            fig, ax = plt.subplots(figsize=(8, 6))
            cluster_counts.plot(kind='bar', ax=ax, color='skyblue')
            plt.title("Sequences per Cluster")
            plt.xlabel("Cluster ID")
            plt.ylabel("Number of Sequences")
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Taxa Abundance")
            fig, ax = plt.subplots(figsize=(8, 6))
            # Show top 10 taxa
            top_taxa = taxa_counts.head(10)
            top_taxa.plot(kind='barh', ax=ax, color='lightcoral')
            plt.title("Top Taxa by Sequence Count")
            plt.xlabel("Number of Sequences")
            st.pyplot(fig)
        
        # Summary Report
        st.header("📋 Analysis Summary")
        
        total_sequences = len(analyzer.sequences)
        known_sequences = len([x for x in analyzer.taxa_labels if x != 'Unknown'])
        novel_sequences = total_sequences - known_sequences
        unique_clusters = len(set(analyzer.clusters))
        
        summary_text = f"""
        **Biodiversity Analysis Report**
        
        📊 **Dataset Overview:**
        - Total sequences analyzed: {total_sequences}
        - Known taxonomic assignments: {known_sequences} ({known_sequences/total_sequences:.1%})
        - Potential novel sequences: {novel_sequences} ({novel_sequences/total_sequences:.1%})
        
        🎯 **Clustering Results:**
        - Number of clusters identified: {unique_clusters}
        - Clustering method: {clustering_method.upper()}
        
        🤖 **Classification Performance:**
        - Model accuracy: {accuracy:.2%} if accuracy else "Insufficient data"
        
        🔬 **Key Findings:**
        - The analysis identified {unique_clusters} distinct sequence clusters
        - {novel_sequences} sequences could represent novel or uncharacterized taxa
        - Most abundant taxon: {taxa_counts.index[0] if len(taxa_counts) > 0 else "N/A"}
        
        💡 **Recommendations for further analysis:**
        - Validate novel clusters with phylogenetic analysis
        - Increase reference database coverage
        - Consider environmental metadata for ecological insights
        """
        
        st.markdown(summary_text)
        
        # Download results
        st.header("💾 Export Results")
        
        results_df = pd.DataFrame({
            'Sequence_ID': range(len(analyzer.sequences)),
            'Taxa': analyzer.taxa_labels,
            'Sequence_Type': analyzer.sequence_types,
            'Cluster': analyzer.clusters,
            'Sequence_Length': [len(seq) for seq in analyzer.sequences]
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Results as CSV",
            data=csv,
            file_name="seaquence_results.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Upload a FASTA file or use sample data to begin analysis")
        
        # Show app information
        st.markdown("""
        ### 🌊 About SeaQuence
        
        SeaQuence is an AI-powered tool for analyzing environmental DNA (eDNA) sequences to:
        
        - 🧬 **Identify sequence types** (DNA vs Protein)
        - 🎯 **Cluster similar sequences** using unsupervised learning
        - 🤖 **Classify known taxa** with machine learning
        - 🔍 **Discover novel biodiversity** through cluster analysis
        - 📊 **Estimate species abundance** from sequence counts
        
        ### 🚀 How to use:
        1. Upload your FASTA file or try the sample data
        2. Adjust analysis parameters in the sidebar
        3. Explore clustering and classification results
        4. Download your results for further analysis
        
        ### 🔬 Technical approach:
        - K-mer tokenization (DNA: k=6, Protein: k=3)
        - Bag-of-k-mers vectorization
        - KMeans/DBSCAN clustering
        - Random Forest classification
        - PCA/t-SNE visualization
        """)

if __name__ == "__main__":
    main()
