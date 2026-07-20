# submission/checklist.py
"""
Comprehensive pre-submission checklist
Miss ANY of these = likely rejection
"""

from pathlib import Path
import subprocess
import json
from typing import List, Dict
import re

class SubmissionChecker:
    """
    Automated checks before submission
    """
    
    def __init__(self, paper_dir: str = './paper'):
        self.paper_dir = Path(paper_dir)
        self.issues = []
        self.warnings = []
        
    def check_all(self) -> Dict:
        """
        Run all checks
        """
        print("[CHECK] Running pre-submission checks...\n")
        
        checks = [
            self.check_paper_formatting,
            self.check_references,
            self.check_figures,
            self.check_tables,
            self.check_reproducibility,
            self.check_statistical_significance,
            self.check_code_release,
            self.check_anonymization,  # For double-blind venues
            self.check_page_limit,
            self.check_supplementary,
        ]
        
        for check_fn in checks:
            check_fn()
        
        # Print results
        self.print_results()
        
        return {
            'issues': self.issues,
            'warnings': self.warnings,
            'ready_to_submit': len(self.issues) == 0
        }
    
    def check_paper_formatting(self):
        """Check LaTeX formatting"""
        print("[PAPER] Checking paper formatting...")
        
        main_tex = self.paper_dir / 'main.tex'
        
        if not main_tex.exists():
            self.issues.append("main.tex not found")
            return
        
        with open(main_tex) as f:
            content = f.read()
        
        # Check document class
        if 'neurips' not in content and 'iclr' not in content:
            self.warnings.append("Using correct conference template?")
        
        # Check for common errors
        if r'\cite{' in content:
            # Good
            pass
        else:
            self.warnings.append("No citations found - check references")
        
        # Check for TODO/FIXME
        if 'TODO' in content or 'FIXME' in content:
            self.issues.append("Remove TODO/FIXME comments")
        
        # Check for proper theorem environments
        if r'\begin{theorem}' in content:
            if r'\end{theorem}' not in content:
                self.issues.append("Unclosed theorem environment")
        
        print("  OK Formatting check complete")
    
    def check_references(self):
        """Check bibliography"""
        print("[REFS] Checking references...")
        
        bib_file = self.paper_dir / 'references.bib'
        
        if not bib_file.exists():
            self.issues.append("references.bib not found")
            return
        
        with open(bib_file) as f:
            bib_content = f.read()
        
        # Count references
        num_refs = bib_content.count('@article') + bib_content.count('@inproceedings')
        
        if num_refs < 20:
            self.warnings.append(f"Only {num_refs} references - add more related work")
        
        # Check for required citations
        required_keywords = [
            'transformer', 'attention', 'spiking', 'neuromorphic',
            'vaswani', 'loihi'
        ]
        
        for keyword in required_keywords:
            if keyword.lower() not in bib_content.lower():
                self.warnings.append(f"Missing citation related to '{keyword}'")
        
        # Check for broken citations
        if 'MISSING' in bib_content or 'TODO' in bib_content:
            self.issues.append("Incomplete bibliography entries")
        
        print(f"  OK Found {num_refs} references")
    
    def check_figures(self):
        """Check all figures"""
        print("[FIG]  Checking figures...")
        
        figures_dir = self.paper_dir / 'figures'
        
        if not figures_dir.exists():
            self.issues.append("figures/ directory not found")
            return
        
        # Check for vector formats
        pdf_figs = list(figures_dir.glob('*.pdf'))
        png_figs = list(figures_dir.glob('*.png'))
        
        if len(png_figs) > 0:
            self.warnings.append(f"{len(png_figs)} PNG figures - convert to PDF for quality")
        
        # Check figure quality
        for fig in pdf_figs:
            # Check file size (too large = problem)
            size_mb = fig.stat().st_size / (1024 * 1024)
            if size_mb > 5:
                self.warnings.append(f"{fig.name} is {size_mb:.1f}MB - compress")
        
        # Check that all referenced figures exist
        main_tex = self.paper_dir / 'main.tex'
        with open(main_tex) as f:
            content = f.read()
        
        # Find \includegraphics commands
        fig_refs = re.findall(r'\\includegraphics.*?\{figures/(.*?)\}', content)
        
        for fig_ref in fig_refs:
            # Remove extension if present
            fig_name = fig_ref.replace('.pdf', '').replace('.png', '')
            fig_path = figures_dir / f"{fig_name}.pdf"
            
            if not fig_path.exists():
                # Try PNG
                fig_path = figures_dir / f"{fig_name}.png"
                if not fig_path.exists():
                    self.issues.append(f"Referenced figure not found: {fig_ref}")
        
        print(f"  OK Found {len(pdf_figs)} PDF figures")
    
    def check_tables(self):
        """Check tables"""
        print("[TABLE] Checking tables...")
        
        tables_dir = self.paper_dir / 'tables'
        
        if not tables_dir.exists():
            self.warnings.append("tables/ directory not found")
            return
        
        # Check for proper formatting
        tex_tables = list(tables_dir.glob('*.tex'))
        
        for table_file in tex_tables:
            with open(table_file) as f:
                content = f.read()
            
            # Check for proper table environment
            if r'\begin{table' not in content:
                self.warnings.append(f"{table_file.name} missing table environment")
            
            # Check for caption and label
            if r'\caption{' not in content:
                self.warnings.append(f"{table_file.name} missing caption")
            
            if r'\label{' not in content:
                self.warnings.append(f"{table_file.name} missing label")
            
            # Check for booktabs
            if 'tabular' in content and 'booktabs' not in content:
                self.warnings.append(f"{table_file.name} should use booktabs package")
        
        print(f"  OK Found {len(tex_tables)} table files")
    
    def check_reproducibility(self):
        """Check reproducibility materials"""
        print("[REPRO] Checking reproducibility...")
        
        # Check for code repository
        if not Path('./README.md').exists():
            self.issues.append("README.md not found - add setup instructions")
        
        # Check for requirements.txt or environment.yml
        if not Path('./requirements.txt').exists() and not Path('./environment.yml').exists():
            self.issues.append("Missing requirements.txt or environment.yml")
        
        # Check for pretrained models
        models_dir = Path('./pretrained_models')
        if not models_dir.exists():
            self.warnings.append("No pretrained models provided")
        
        # Check for experiment scripts
        scripts = list(Path('./scripts').glob('*.py')) if Path('./scripts').exists() else []
        if len(scripts) == 0:
            self.warnings.append("No experiment scripts found")
        
        print("  OK Reproducibility check complete")
    
    def check_statistical_significance(self):
        """Check for statistical tests"""
        print("[STAT] Checking statistical significance...")
        
        main_tex = self.paper_dir / 'main.tex'
        with open(main_tex) as f:
            content = f.read()
        
        # Look for statistical terminology
        stat_terms = ['p-value', 'p <', 'statistically significant', 't-test', 'confidence interval']
        
        found_stats = any(term in content for term in stat_terms)
        
        if not found_stats:
            self.issues.append("No statistical significance testing mentioned")
        
        # Check for multiple seeds
        if 'seed' not in content.lower() and 'random' not in content.lower():
            self.issues.append("No mention of multiple random seeds")
        
        print("  OK Statistical checks complete")
    
    def check_code_release(self):
        """Check code release preparation"""
        print("[CODE] Checking code release...")
        
        # Check for .gitignore
        if not Path('./.gitignore').exists():
            self.warnings.append("No .gitignore file")
        
        # Check for LICENSE
        if not Path('./LICENSE').exists():
            self.issues.append("No LICENSE file - add MIT or Apache 2.0")
        
        # Check for sensitive information
        sensitive_patterns = [
            'api_key', 'password', 'secret', 'token', 'private_key'
        ]
        
        for py_file in Path('.').rglob('*.py'):
            with open(py_file) as f:
                content = f.read().lower()
            
            for pattern in sensitive_patterns:
                if pattern in content:
                    self.warnings.append(f"Possible sensitive info in {py_file}")
        
        print("  OK Code release check complete")
    
    def check_anonymization(self):
        """Check for double-blind compliance"""
        print("[ANON] Checking anonymization (for double-blind venues)...")
        
        main_tex = self.paper_dir / 'main.tex'
        with open(main_tex) as f:
            content = f.read()
        
        # Check for author information
        if r'\author{' in content:
            # Check if using \iclrfinalcopy or similar
            if 'final' not in content.lower():
                self.warnings.append("Author information present - anonymize for initial submission")
        
        # Check for institutional references
        suspicious = ['our university', 'our institution', 'our lab']
        for phrase in suspicious:
            if phrase in content.lower():
                self.warnings.append(f"Possible de-anonymization: '{phrase}'")
        
        # Check for self-citations that reveal identity
        if 'our previous work' in content.lower() or 'we previously' in content.lower():
            self.warnings.append("Self-citations may reveal identity - use third person")
        
        print("  OK Anonymization check complete")
    
    def check_page_limit(self):
        """Check page count"""
        print("[PAGE] Checking page limit...")
        
        # Compile LaTeX and check pages
        main_tex = self.paper_dir / 'main.tex'
        
        try:
            # Try to compile
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
                cwd=self.paper_dir,
                capture_output=True,
                timeout=30
            )
            
            # Get page count from PDF
            pdf_file = self.paper_dir / 'main.pdf'
            if pdf_file.exists():
                # Use PyPDF2 to count pages
                try:
                    import PyPDF2
                    with open(pdf_file, 'rb') as f:
                        pdf = PyPDF2.PdfReader(f)
                        num_pages = len(pdf.pages)
                    
                    print(f"  [PAPER] Main paper: {num_pages} pages")
                    
                    # NeurIPS/ICLR typically allow 8-9 pages + references
                    if num_pages > 9:
                        self.warnings.append(f"Paper is {num_pages} pages - may exceed limit")
                except ImportError:
                    self.warnings.append("Install PyPDF2 to check page count")
        
        except subprocess.TimeoutExpired:
            self.warnings.append("LaTeX compilation timed out")
        except Exception as e:
            self.warnings.append(f"Could not compile LaTeX: {e}")
        
        print("  OK Page limit check complete")
    
    def check_supplementary(self):
        """Check supplementary material"""
        print("[SUPP] Checking supplementary material...")
        
        supp_file = self.paper_dir / 'supplementary.pdf'
        
        if supp_file.exists():
            print("  OK Supplementary material found")
        else:
            self.warnings.append("No supplementary material - consider adding proofs/extra results")
        
        # Check appendix
        main_tex = self.paper_dir / 'main.tex'
        with open(main_tex) as f:
            content = f.read()
        
        if r'\appendix' in content:
            print("  OK Appendix section found")
        else:
            self.warnings.append("No appendix - add detailed proofs/ablations")
    
    def print_results(self):
        """Print check results"""
        print("\n" + "="*60)
        print("SUBMISSION READINESS REPORT")
        print("="*60)
        
        if len(self.issues) == 0:
            print("\n[OK] NO CRITICAL ISSUES FOUND!")
        else:
            print(f"\n[FAIL] {len(self.issues)} CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  - {issue}")
        
        if len(self.warnings) > 0:
            print(f"\n[WARN]  {len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if len(self.issues) == 0 and len(self.warnings) == 0:
            print("\n[DONE] PAPER IS READY TO SUBMIT!")
        elif len(self.issues) == 0:
            print("\n[OK] Paper is submittable, but address warnings for best results")
        else:
            print("\n[FAIL] FIX CRITICAL ISSUES BEFORE SUBMITTING")
        
        print("="*60)


# Run checks
if __name__ == '__main__':
    checker = SubmissionChecker()
    results = checker.check_all()
    
    # Save report
    with open('submission_checklist.json', 'w') as f:
        json.dump(results, f, indent=2)