import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class ReceiptItem:
    name: str
    quantity: int = 1
    unit_price: float = 0.0
    total_price: float = 0.0
    discount: float = 0.0
    final_price: float = 0.0

@dataclass
class ReceiptData:
    items: List[ReceiptItem] = field(default_factory=list)
    subtotal: float = 0.0
    total_discount: float = 0.0
    final_total: float = 0.0
    payment_method: str = ""
    amount_paid: float = 0.0
    change_given: float = 0.0

class ReceiptParser:
    """
    Parses cleaned receipt text and extracts structured data
    """
    
    def __init__(self):
        self.price_pattern = re.compile(r'[£$€]?(\d+\.\d{2})')
        self.quantity_pattern = re.compile(r'(\d+)x\s*(.+)', re.IGNORECASE)
        self.discount_keywords = ['discount', 'override', 'reduction', 'off', 'save']
        self.payment_methods = ['cash', 'card', 'credit', 'debit', 'contactless', 'chip', 'pin']
    
    def parse_receipt(self, cleaned_text: str) -> ReceiptData:
        """Parse cleaned receipt text into structured data"""
        lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        receipt = ReceiptData()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip header/store info (usually first few lines without prices)
            if not self._contains_price(line) and i < 3:
                i += 1
                continue
            
            # Parse items with quantities and prices
            if self._is_item_line(line):
                item = self._parse_item_line(line)
                if item:
                    # Check if next line is a discount for this item
                    if i + 1 < len(lines) and self._is_discount_line(lines[i + 1]):
                        discount_amount = self._extract_price(lines[i + 1])
                        if discount_amount:
                            item.discount = abs(discount_amount)
                            item.final_price = item.total_price - item.discount
                        i += 1  # Skip the discount line since we processed it
                    else:
                        item.final_price = item.total_price
                    receipt.items.append(item)
            
            # Parse totals (skip individual discount lines as they're handled above)
            elif self._is_subtotal_line(line):
                receipt.subtotal = self._extract_price(line) or 0.0
            
            elif self._is_total_discount_line(line):
                receipt.total_discount = abs(self._extract_price(line) or 0.0)
            
            elif self._is_final_total_line(line):
                receipt.final_total = self._extract_price(line) or 0.0
            
            # Parse payment info
            elif self._is_payment_line(line):
                payment_info = self._parse_payment_line(line)
                if payment_info:
                    receipt.payment_method = payment_info[0]
                    receipt.amount_paid = payment_info[1]
            
            elif self._is_change_line(line):
                receipt.change_given = self._extract_price(line) or 0.0
            
            i += 1
        
        return receipt
    
    def _contains_price(self, line: str) -> bool:
        """Check if line contains a price"""
        return bool(self.price_pattern.search(line))
    
    def _extract_price(self, line: str) -> Optional[float]:
        """Extract price from line, handling negative values"""
        match = self.price_pattern.search(line)
        if match:
            price = float(match.group(1))
            # Check if price is negative
            if '-' in line.split(match.group(0))[0]:
                price = -price
            return price
        return None
    
    def _is_item_line(self, line: str) -> bool:
        """Determine if line represents an item purchase"""
        line_lower = line.lower()
        
        # Skip lines that are clearly not items
        skip_keywords = ['total', 'discount', 'change', 'cash', 'card', 'receipt', 'thank', 'admin', 'manager']
        if any(keyword in line_lower for keyword in skip_keywords):
            return False
        
        # Must contain a price and some text
        has_price = self._contains_price(line)
        has_text = len(line.replace('£', '').replace('$', '').replace('€', '').strip()) > 3
        
        # Likely an item if it has quantity pattern or price at end
        has_quantity = bool(self.quantity_pattern.match(line))
        price_at_end = line.strip().endswith(('0', '5'))  # Prices usually end in 0 or 5 cents
        
        return has_price and has_text and (has_quantity or price_at_end)
    
    def _parse_item_line(self, line: str) -> Optional[ReceiptItem]:
        """Parse a line containing an item"""
        # Check for quantity pattern first
        qty_match = self.quantity_pattern.match(line.strip())
        if qty_match:
            quantity = int(qty_match.group(1))
            remainder = qty_match.group(2).strip()
        else:
            quantity = 1
            remainder = line.strip()
        
        # Extract price (usually at the end)
        price = self._extract_price(remainder)
        if not price:
            return None
        
        # Extract item name (everything except the price)
        price_str = self.price_pattern.search(remainder).group(0)
        name = remainder.replace(price_str, '').strip()
        
        # Clean up name
        name = re.sub(r'^[\d\s\*\-]+', '', name).strip()  # Remove leading numbers/symbols
        name = re.sub(r'\s+', ' ', name)  # Normalize spaces
        
        if not name:
            return None
        
        return ReceiptItem(
            name=name,
            quantity=quantity,
            unit_price=price / quantity if quantity > 0 else price,
            total_price=price,
            final_price=price  # Will be updated if discount is found
        )
    
    def _is_discount_line(self, line: str) -> bool:
        """Check if line represents a discount"""
        line_lower = line.lower()
        return any(keyword in line_lower for keyword in self.discount_keywords) and self._contains_price(line)
    
    def _is_subtotal_line(self, line: str) -> bool:
        """Check if line represents subtotal"""
        return 'sub' in line.lower() and 'total' in line.lower() and self._contains_price(line)
    
    def _is_total_discount_line(self, line: str) -> bool:
        """Check if line represents total discount amount"""
        line_lower = line.lower()
        return ('discount' in line_lower and 'total' in line_lower) and self._contains_price(line)
    
    def _is_final_total_line(self, line: str) -> bool:
        """Check if line represents final total"""
        line_lower = line.lower()
        # Look for "Total:" but not "Sub Total:" or "Discount Total:"
        return (line_lower.startswith('total') and self._contains_price(line) and 
                'sub' not in line_lower and 'discount' not in line_lower)
    
    def _is_payment_line(self, line: str) -> bool:
        """Check if line represents payment method/amount"""
        line_lower = line.lower()
        return any(method in line_lower for method in self.payment_methods) and self._contains_price(line)
    
    def _parse_payment_line(self, line: str) -> Optional[Tuple[str, float]]:
        """Extract payment method and amount"""
        line_lower = line.lower()
        method = next((m for m in self.payment_methods if m in line_lower), "unknown")
        amount = self._extract_price(line)
        return (method, amount) if amount else None
    
    def _is_change_line(self, line: str) -> bool:
        """Check if line represents change given"""
        return 'change' in line.lower() and self._contains_price(line)
    
    def format_receipt_data(self, receipt: ReceiptData) -> str:
        """Format parsed receipt data for display"""
        output = []
        output.append("=== PARSED RECEIPT DATA ===\n")
        
        # Items
        output.append("ITEMS PURCHASED:")
        for i, item in enumerate(receipt.items, 1):
            output.append(f"{i}. {item.name}")
            output.append(f"   Quantity: {item.quantity}")
            output.append(f"   Unit Price: £{item.unit_price:.2f}")
            output.append(f"   Total Price: £{item.total_price:.2f}")
            if item.discount > 0:
                output.append(f"   Discount: -£{item.discount:.2f}")
                output.append(f"   Final Price: £{item.final_price:.2f}")
            else:
                output.append(f"   Final Price: £{item.final_price:.2f}")
            output.append("")
        
        # Summary
        output.append("SUMMARY:")
        if receipt.subtotal > 0:
            output.append(f"Subtotal: £{receipt.subtotal:.2f}")
        if receipt.total_discount > 0:
            output.append(f"Total Discounts: -£{receipt.total_discount:.2f}")
        output.append(f"Final Total: £{receipt.final_total:.2f}")
        
        # Payment
        output.append("\nPAYMENT:")
        if receipt.payment_method:
            output.append(f"Method: {receipt.payment_method.title()}")
            output.append(f"Amount Paid: £{receipt.amount_paid:.2f}")
        if receipt.change_given > 0:
            output.append(f"Change Given: £{receipt.change_given:.2f}")
        
        return "\n".join(output)