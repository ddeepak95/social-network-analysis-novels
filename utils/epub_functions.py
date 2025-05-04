from typing import Optional, List, Dict, Any
import os
import json
import uuid
import re
from bs4 import BeautifulSoup
import html2text

from ebooklib import epub

def extract_epub_files(epub_path: str, working_folder: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """Process the epub file and extract its contents
    
    Args:
        epub_path: Path to the epub file
        working_folder: Directory to extract files to
        doc_id: Optional document identifier for the epub. If None, will use filename or generate a UUID.
        
    Returns:
        Dictionary containing TOC and file details
    """
    book = epub.read_epub(epub_path)
    
    # Generate doc_id if not provided
    if doc_id is None:
        # Use the epub filename without extension or generate a UUID if unable to determine
        try:
            doc_id = os.path.splitext(os.path.basename(epub_path))[0]
        except:
            doc_id = str(uuid.uuid4())
    
    # Extract TOC
    toc = get_toc_details(book)

    # Extract contents
    output_dir = os.path.join(working_folder, "unbundled_epub")
    os.makedirs(output_dir, exist_ok=True)
    
    file_details = []
    
    for item in book.get_items():
        if item.file_name:
            file_path = os.path.join(output_dir, item.file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(item.content)
                
            file_details.append({
                'file_name': os.path.basename(item.file_name),
                'file_ext': os.path.splitext(item.file_name)[1][1:],
                'file_path': item.file_name
            })
    
    # Save TOC to JSON
    toc_json_path = os.path.join(working_folder, f"{doc_id}_toc.json")
    with open(toc_json_path, 'w', encoding='utf-8') as f:
        json.dump(toc, f, indent=2)
    
    return {
        "doc_id": doc_id,
        "toc": toc,
        "file_details": file_details,
        "output_dir": output_dir,
        "toc_json_path": toc_json_path
    }

def get_toc_details(book: epub.EpubBook) -> list:
    """Extract table of contents details from epub book"""
    toc = book.toc
    toc_details = []
    
    for item in toc:
        if isinstance(item, epub.Link):
            toc_item = {
                "type": "link",
                "href": item.href,
                "title": item.title,
                "uid": item.uid
            }
            toc_details.append(toc_item)
        elif isinstance(item, tuple) and isinstance(item[0], epub.Section):
            toc_item = {
                "type": "section",
                "title": item[0].title,
                "links": []
            }
            for link in item[1]:
                toc_item["links"].append({
                    "type": "link",
                    "href": link.href,
                    "title": link.title,
                    "uid": link.uid
                })
            toc_details.append(toc_item)
            
    return toc_details

def get_file_content_by_toc_reference(toc_reference: Dict, working_folder: str) -> str:
    """Get content of a file referenced in the TOC
    
    Args:
        toc_reference: Dictionary containing href from TOC
        working_folder: Directory where files were extracted
        
    Returns:
        Content of the referenced file as string
    """
    if "href" not in toc_reference:
        raise ValueError("TOC reference must contain 'href' key")
    
    # Remove anchor part (everything after #) from the href
    href = toc_reference["href"].split('#')[0]
    
    file_path = os.path.join(working_folder, "unbundled_epub", href)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Some EPUB content files might be binary (images, etc.)
        # or use different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            return f"[Binary content or unsupported encoding in {href}]"

def create_flattened_book_json(epub_path: str, output_json_path: str, doc_id: Optional[str] = None):
    """
    Create a single flattened JSON file containing all book content organized by sections.
    
    Args:
        epub_path: Path to the epub file
        output_json_path: Path where the JSON output will be saved
        doc_id: Optional document identifier
        
    Returns:
        Path to the created JSON file
    """
    # Create a temporary working directory
    temp_dir = os.path.join(os.path.dirname(output_json_path), "temp_extraction")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract the epub
    result = extract_epub_files(epub_path, temp_dir, doc_id)
    
    # Create flattened content structure
    flattened_content = []
    
    # First try to extract content using TOC
    for item in result["toc"]:
        if item["type"] == "link":
            title = item["title"]
            try:
                content = get_file_content_by_toc_reference(item, temp_dir)
                
                flattened_content.append({
                    "title": title,
                    "content": content
                })
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: {e}")
                # We'll handle missing files in the fallback approach below
                
        elif item["type"] == "section":
            section_title = item["title"]
            
            # Handle links within sections
            for link in item.get("links", []):
                combined_title = f"{section_title} - {link['title']}"
                try:
                    content = get_file_content_by_toc_reference(link, temp_dir)
                    
                    flattened_content.append({
                        "title": combined_title,
                        "content": content
                    })
                except (FileNotFoundError, ValueError) as e:
                    print(f"Warning: {e}")
                    # We'll handle missing files in the fallback approach below
    
    # If we didn't get any content from TOC approach, try a fallback
    if not flattened_content or all(not item.get('content') for item in flattened_content):
        print("Falling back to direct file extraction...")
        
        # Get all HTML/XHTML files in order
        html_files = []
        output_dir = os.path.join(temp_dir, "unbundled_epub")
        
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.html', '.xhtml', '.htm')):
                    html_files.append(os.path.join(root, file))
        
        # Sort files to maintain chapter order (often numeric prefixes help)
        html_files.sort()
        
        # Extract content from each file
        for i, file_path in enumerate(html_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use filename as title if nothing better
                title = f"Chapter {i+1}"
                rel_path = os.path.relpath(file_path, output_dir)
                
                flattened_content.append({
                    "title": title,
                    "content": content,
                    "file_path": rel_path
                })
            except UnicodeDecodeError:
                # Try another encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    
                    title = f"Chapter {i+1}"
                    rel_path = os.path.relpath(file_path, output_dir)
                    
                    flattened_content.append({
                        "title": title,
                        "content": content,
                        "file_path": rel_path
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(flattened_content, f, indent=2, ensure_ascii=False)
    
    # Optionally clean up the temporary directory
    # import shutil
    # shutil.rmtree(temp_dir)
    
    return output_json_path

def html_to_markdown(html_content: str, strip_images: bool = True, strip_tables: bool = False) -> str:
    """
    Convert HTML content to Markdown format suitable for LLM prompting.
    
    Args:
        html_content: The HTML content to convert
        strip_images: Whether to remove images from the content (default True)
        strip_tables: Whether to remove tables from the content (default False)
        
    Returns:
        Cleaned markdown text
    """
    # First clean up the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Optionally remove images
    if strip_images:
        for img in soup.find_all('img'):
            img.extract()
    
    # Optionally remove tables
    if strip_tables:
        for table in soup.find_all('table'):
            table.extract()
    
    # Configure html2text
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = True
    h2t.ignore_tables = False
    h2t.ignore_emphasis = False
    h2t.body_width = 0  # No wrapping
    
    # Convert to markdown
    markdown = h2t.handle(str(soup))
    
    # Post-processing cleanup
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excessive newlines
    markdown = re.sub(r'\[!\[[^\]]*\]\([^\)]*\)\]\([^\)]*\)', '', markdown)  # Remove nested image links
    
    return markdown.strip()