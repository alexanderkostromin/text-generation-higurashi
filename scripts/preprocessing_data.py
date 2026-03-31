import re


def merge_characters_lines(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        current_speaker = None
        current_text = []
        is_first_content = True

        for line in f_in:
            raw_line = line.strip()

            if not raw_line:
                if current_speaker:
                    f_out.write(f"{current_speaker}: {' '.join(current_text)}\n")
                    current_speaker = None
                    current_text = []

                if not is_first_content:
                    f_out.write("\n")
                continue

            episode_match = re.search(r'\d+\s+серия', raw_line, re.IGNORECASE)
            if episode_match:
                if current_speaker:
                    f_out.write(f"{current_speaker}: {' '.join(current_text)}\n")
                    current_speaker = None
                    current_text = []

                if not is_first_content:
                    f_out.write("\n")

                f_out.write(f"--- {raw_line} ---\n")
                is_first_content = False
                continue

            match = re.match(r'^([^:\n]+):\s*(.*)', raw_line)
            if match:
                is_first_content = False
                speaker, text = match.group(1).strip(), match.group(2).strip()

                if speaker == current_speaker:
                    current_text.append(text)
                else:
                    if current_speaker:
                        f_out.write(f"{current_speaker}: {' '.join(current_text)}\n")
                    current_speaker, current_text = speaker, [text]
            else:
                if current_speaker:
                    current_text.append(raw_line)

        if current_speaker:
            f_out.write(f"{current_speaker}: {' '.join(current_text)}\n")

    print(f"Обработка завершена. Файл сохранен: {output_path}")


def clean_and_structure_for_dataset(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        is_start = True

        for line in f_in:
            if line.startswith("---"):
                if not is_start:
                    f_out.write("\n\n\n\n")
                continue

            if not line.strip():
                if not is_start:
                    f_out.write("\n")
                continue

            clean_line = re.sub(r'^[^:\n]+:\s*', '', line).strip()

            if clean_line:
                f_out.write(clean_line + "\n")
                is_start = False

    print(f"Обработка завершена. Файл сохранен: {output_path}")


if __name__ == '__main__':
    input_path = '../data/higurashi_no_naku_koro_ni_tv_subtitles_ru/hand_processed/'\
        'hand_processed_subtitles.txt'
    output_path = '../data/higurashi_no_naku_koro_ni_tv_subtitles_ru/processed/'\
        'higurashi_merged.txt'
    merge_characters_lines(input_path, output_path)

    input_path = '../data/higurashi_no_naku_koro_ni_tv_subtitles_ru/processed/'\
        'higurashi_merged.txt'
    output_path = '../data/higurashi_no_naku_koro_ni_tv_subtitles_ru/processed/'\
        'higurashi_clean.txt'
    clean_and_structure_for_dataset(input_path, output_path)
