#!/usr/bin/env python3

from typing import List
from compressors import SearchResult

def create_sample_documents() -> List[SearchResult]:
    """Create sample documents for demonstration."""
    documents = [
        SearchResult(
            evi_id=1,
            docid=1,
            title="Solid State Drives Overview",
            text="""Solid-state drives (SSDs) are storage devices that use flash memory to store data persistently. 
            Unlike traditional hard disk drives (HDDs), SSDs have no moving mechanical parts, which makes them more 
            reliable and faster. SSDs use NAND flash memory cells to store data electronically.""",
            score=0.95
        ),
        SearchResult(
            evi_id=2,
            docid=2,
            title="SSD Performance Benefits",
            text="""SSDs provide significant performance improvements over traditional hard drives. They offer faster 
            boot times, quicker application loading, and improved overall system responsiveness. The access time for 
            SSDs is typically under 1 millisecond, compared to 5-10 milliseconds for HDDs.""",
            score=0.90
        ),
        SearchResult(
            evi_id=3,
            docid=3,
            title="SSD Technology Details",
            text="""Modern SSDs use various types of NAND flash memory, including SLC, MLC, TLC, and QLC. 
            They connect to computers via interfaces like SATA, PCIe, or M.2. SSDs consume less power than HDDs 
            and generate less heat, making them ideal for laptops and mobile devices.""",
            score=0.85
        ),
        SearchResult(
            evi_id=4,
            docid=4,
            title="Computing History",
            text="""The first computers were built in the 1940s and used vacuum tubes. Personal computers became 
            popular in the 1980s. The internet was invented in the late 20th century. Today, we have smartphones 
            and cloud computing. Weather prediction has also improved significantly.""",
            score=0.30
        ),
        SearchResult(
            evi_id=5,
            docid=5,
            title="SSD vs HDD Comparison",
            text="""When comparing SSDs to HDDs, SSDs win in speed, reliability, and power consumption. However, 
            HDDs typically offer more storage capacity per dollar. SSDs have become much more affordable in recent 
            years, making them the preferred choice for most users seeking better performance.""",
            score=0.88
        )
    ]
    return documents
