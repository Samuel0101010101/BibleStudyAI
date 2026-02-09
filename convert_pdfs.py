"""
Convert all PDFs in raw files/ to text files in sources/
"""
import os
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Installing pypdf...")
    os.system("pip install -q pypdf")
    from pypdf import PdfReader

def convert_pdf_to_text(pdf_path, output_path):
    """Extract text from PDF and save to text file"""
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        
        # Extract text from all pages
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_content.append(text)
        
        # Write to text file
        full_text = "\n\n".join(text_content)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"SOURCE: {pdf_path.name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(full_text)
        
        return len(reader.pages), len(full_text)
    except Exception as e:
        return None, str(e)

def main():
    raw_dir = Path("raw files")
    sources_dir = Path("sources")
    sources_dir.mkdir(exist_ok=True)
    
    # Get all PDFs
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    total = len(pdf_files)
    
    print(f"\n📚 Converting {total} PDFs to text files...\n")
    
    success_count = 0
    failed_files = []
    total_chars = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        # Create text filename (keep original name, change extension)
        txt_filename = pdf_path.stem + ".txt"
        output_path = sources_dir / txt_filename
        
        # Skip if already converted
        if output_path.exists():
            print(f"[{idx}/{total}] {pdf_path.name[:60]}... ⏭️  (already exists)")
            success_count += 1
            continue
        
        print(f"[{idx}/{total}] {pdf_path.name[:60]}...", end=" ", flush=True)
        
        try:
            pages, result = convert_pdf_to_text(pdf_path, output_path)
            
            if pages is not None:
                print(f"✅ ({pages} pages, {result:,} chars)")
                success_count += 1
                total_chars += result
            else:
                print(f"❌ Error: {result}")
                failed_files.append((pdf_path.name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Conversion interrupted by user")
            break
        except Exception as e:
            print(f"❌ Crash: {str(e)[:50]}")
            failed_files.append((pdf_path.name, f"Crash: {e}"))
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully converted: {success_count}/{total}")
    print(f"📊 Total text extracted: {total_chars:,} characters ({total_chars/1024/1024:.1f} MB)")
    
    if failed_files:
        print(f"\n❌ Failed files ({len(failed_files)}):")
        for name, error in failed_files[:10]:  # Show first 10
            print(f"   - {name}: {error}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")
    
    print(f"\n💾 Text files saved to: {sources_dir.absolute()}")
    print(f"🔄 Next: Restart bot to rebuild vector database with new sources\n")

if __name__ == "__main__":
    main()
