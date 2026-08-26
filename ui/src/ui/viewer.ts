/** Side panel for the read-only opening-book viewer: Prev/Next buttons plus
 * a scrollable no.1..no.N list; clicking an item (or arrow keys) selects it. */
export class OpenStateViewer {
  private prevBtn: HTMLButtonElement;
  private nextBtn: HTMLButtonElement;
  private items: HTMLButtonElement[] = [];
  private selected = -1;

  constructor(root: HTMLElement, count: number, private onSelect: (index: number) => void) {
    const section = document.createElement('section');
    section.className = 'panel-section';

    const h2 = document.createElement('h2');
    h2.textContent = 'Open states';
    section.className = 'panel-section openbook-section';

    const btnRow = document.createElement('div');
    btnRow.className = 'btn-row';
    this.prevBtn = document.createElement('button');
    this.prevBtn.textContent = '◀ Prev';
    this.nextBtn = document.createElement('button');
    this.nextBtn.textContent = 'Next ▶';
    this.prevBtn.addEventListener('click', () => this.step(-1));
    this.nextBtn.addEventListener('click', () => this.step(1));
    btnRow.append(this.prevBtn, this.nextBtn);
    section.appendChild(btnRow);

    const list = document.createElement('div');
    list.className = 'openbook-list';
    for (let i = 0; i < count; i++) {
      const item = document.createElement('button');
      item.textContent = `no.${i + 1}`;
      item.addEventListener('click', () => this.select(i));
      list.appendChild(item);
      this.items.push(item);
    }
    section.appendChild(list);
    root.replaceChildren(section);

    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') this.step(-1);
      else if (e.key === 'ArrowRight') this.step(1);
    });
  }

  /** Moves the selection by `delta`, clamped to the list bounds. */
  step(delta: number): void {
    if (this.selected < 0) return;
    this.select(Math.min(this.items.length - 1, Math.max(0, this.selected + delta)));
  }

  select(index: number): void {
    if (index === this.selected) return;
    if (this.selected >= 0) this.items[this.selected]!.classList.remove('selected');
    this.selected = index;
    const item = this.items[index]!;
    item.classList.add('selected');
    item.scrollIntoView({ block: 'nearest' });
    this.prevBtn.disabled = index === 0;
    this.nextBtn.disabled = index === this.items.length - 1;
    this.onSelect(index);
  }
}
