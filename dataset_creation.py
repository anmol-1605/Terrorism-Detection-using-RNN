import pandas as pd
import numpy as np
import random
import re
from sklearn.utils import shuffle

class ImprovedTerrorismDatasetCreator:
    def __init__(self):
        # More diverse and realistic base texts
        self.safe_texts = [
            # Technology & Science
            "Researchers develop new vaccine to combat seasonal flu variants",
            "Solar panel efficiency reaches record high in laboratory tests",
            "Machine learning helps doctors diagnose rare diseases more accurately",
            "Space mission discovers water on distant exoplanet surface",
            "Electric vehicle sales increase due to government incentives",
            "Scientists study coral reef restoration techniques in marine biology",
            "Quantum computer breakthrough enables faster drug discovery research",
            "Agricultural technology improves crop yields for farmers worldwide",
            "Medical device innovation helps patients with mobility disabilities",
            "Renewable energy storage solutions reduce electricity costs significantly",

            # Business & Economy
            "Local restaurant chain expands to five new cities this quarter",
            "Small business owners receive loans for post-pandemic recovery efforts",
            "Real estate market shows steady growth in suburban areas",
            "Tech startup raises funding for mobile app development project",
            "Manufacturing jobs return to region due to infrastructure investments",
            "Tourism industry rebounds with international travel restrictions lifted",
            "E-commerce platform launches new features for online sellers",
            "Financial advisors recommend diversified investment portfolios for retirement",
            "Construction company wins contract for affordable housing development",
            "Banking sector introduces digital payment solutions for customers",

            # Education & Culture
            "University offers new scholarships for students studying environmental science",
            "Public library expands digital resources and community programs",
            "Art museum features contemporary works from emerging local artists",
            "High school implements innovative STEM curriculum for students",
            "Community college provides job training for healthcare workers",
            "Music festival showcases diverse performers from around the world",
            "Cultural center celebrates heritage month with traditional dance performances",
            "Adult education program helps residents earn high school diplomas",
            "Youth sports league promotes teamwork and healthy competition",
            "Book club discusses latest literary fiction and non-fiction titles"
        ]

        self.suspicious_texts = [
            # Legitimate but concerning activities
            "Protesters gather at city hall demanding government transparency",
            "Activists organize peaceful march for climate action this weekend",
            "Citizens question mayor controversial spending decisions at town meeting",
            "Environmental group stages demonstration outside chemical plant facility",
            "Workers union plans strike action over unsafe working conditions",
            "Community organizers petition against proposed highway construction project",
            "Journalists investigate allegations of corruption in local government contracts",
            "Civil rights lawyers challenge discriminatory housing policies in court",
            "Students organize campus rally supporting faculty tenure rights",
            "Residents oppose casino development near elementary school location",

            # Surveillance and privacy concerns
            "Privacy advocates question new facial recognition cameras downtown",
            "Technology company faces criticism over personal data collection practices",
            "Government agency expands surveillance program without public notification",
            "Cybersecurity experts warn about vulnerabilities in voting machines",
            "Phone company provides customer location data to law enforcement",
            "Social media platform faces scrutiny over content moderation policies",
            "Internet service provider blocks access to certain websites",
            "Digital rights group files lawsuit against online censorship practices",
            "Encrypted messaging app removes features after government pressure",
            "Whistleblower reveals extent of corporate espionage activities"
        ]

        self.dangerous_texts = [
            # High-risk terrorism-related content
            "Extremist group recruits followers through encrypted social media channels",
            "Intelligence agencies monitor suspected terrorist cell planning coordinated attacks",
            "Bomb threat forces evacuation of government building downtown",
            "Airport security detects suspicious package during routine screening",
            "Chemical weapons investigation launched after laboratory break-in incident",
            "International terrorist organization claims responsibility for recent bombing",
            "Authorities arrest suspects planning attack on public transportation system",
            "Border patrol intercepts shipment containing illegal weapons and explosives",
            "Counter-terrorism unit conducts raid on suspected safe house",
            "Homeland security raises threat level following credible intelligence reports",

            # Direct violent threats
            "Anonymous online post threatens violence against specific ethnic community",
            "Social media account shares instructions for creating improvised explosive devices",
            "Radical group publishes manifesto calling for armed uprising",
            "Forum discussion details plans for targeting crowded public venues",
            "Encrypted channel coordinates simultaneous attacks across multiple cities",
            "Video message incites followers to commit acts of domestic terrorism",
            "Website provides tactical guidance for evading law enforcement detection",
            "Message board users plan disruption of critical infrastructure systems",
            "Group threatens biological attack on public water supply systems",
            "Online community shares techniques for avoiding security screening procedures"
        ]

    def add_realistic_variations(self, texts, variation_count=3):
        """Add realistic variations to base texts"""
        variations = []

        # Prefixes that add context without changing meaning significantly
        prefixes = [
            "According to recent reports, ",
            "Local news sources indicate that ",
            "Officials have confirmed that ",
            "Witnesses reported that ",
            "Evidence suggests that ",
            "Investigation reveals that ",
            "Sources close to the matter state that ",
            "Multiple agencies report that ",
            "Breaking news confirms that ",
            "Latest updates show that "
        ]

        # Time modifiers
        time_mods = [
            "yesterday", "this morning", "last week", "recently", 
            "over the weekend", "during the investigation", 
            "in the past month", "following the incident"
        ]

        # Location modifiers  
        location_mods = [
            "in the downtown area", "near the city center", "at a local facility",
            "in the metropolitan region", "across the state", "in neighboring districts",
            "at an undisclosed location", "in a residential area"
        ]

        for text in texts:
            variations.append(text)  # Original

            for _ in range(variation_count):
                modified_text = text

                # Randomly apply modifications
                if random.random() < 0.4:
                    prefix = random.choice(prefixes)
                    modified_text = prefix + modified_text.lower()

                if random.random() < 0.3:
                    time_mod = random.choice(time_mods)
                    modified_text = modified_text + f" {time_mod}"

                if random.random() < 0.3:
                    location_mod = random.choice(location_mods)
                    # Insert location modifier at appropriate position
                    words = modified_text.split()
                    insert_pos = min(len(words)//2, len(words)-1)
                    words.insert(insert_pos, location_mod)
                    modified_text = ' '.join(words)

                variations.append(modified_text)

        return variations

    def create_balanced_dataset(self, total_size=5000):
        """Create a balanced dataset with proper distribution"""
        print(f"Creating balanced dataset with {total_size} samples...")

        # Calculate distribution - more balanced approach
        safe_count = int(total_size * 0.4)     # 40% safe
        suspicious_count = int(total_size * 0.35)  # 35% suspicious
        dangerous_count = total_size - safe_count - suspicious_count  # 25% dangerous

        print(f"Target distribution: Safe={safe_count}, Suspicious={suspicious_count}, Dangerous={dangerous_count}")

        # Generate variations
        safe_expanded = self.add_realistic_variations(self.safe_texts, 4)
        suspicious_expanded = self.add_realistic_variations(self.suspicious_texts, 4)  
        dangerous_expanded = self.add_realistic_variations(self.dangerous_texts, 4)

        # Sample to exact counts needed
        safe_final = self.sample_with_repetition(safe_expanded, safe_count)
        suspicious_final = self.sample_with_repetition(suspicious_expanded, suspicious_count)
        dangerous_final = self.sample_with_repetition(dangerous_expanded, dangerous_count)

        # Combine all data
        all_texts = safe_final + suspicious_final + dangerous_final
        all_labels = ([0] * len(safe_final) + 
                     [1] * len(suspicious_final) + 
                     [2] * len(dangerous_final))
        all_categories = (['Safe'] * len(safe_final) + 
                         ['Suspicious'] * len(suspicious_final) + 
                         ['Dangerous'] * len(dangerous_final))

        # Create DataFrame
        df = pd.DataFrame({
            'text': all_texts,
            'label': all_labels,
            'category': all_categories,
            'text_length': [len(text) for text in all_texts],
            'word_count': [len(text.split()) for text in all_texts]
        })

        # Shuffle the dataset thoroughly
        df = shuffle(df, random_state=42).reset_index(drop=True)

        return df

    def sample_with_repetition(self, source_list, target_count):
        """Sample from source list with intelligent repetition"""
        if len(source_list) >= target_count:
            return random.sample(source_list, target_count)

        result = source_list.copy()
        remaining = target_count - len(source_list)

        # Add variations by repeating and slightly modifying
        for _ in range(remaining):
            base_text = random.choice(source_list)

            # Minor modifications to avoid exact duplicates
            modifications = [
                lambda x: x.replace("the", "a") if "the" in x else x,
                lambda x: x.replace("is", "was") if "is" in x else x,
                lambda x: x.replace("will", "may") if "will" in x else x,
                lambda x: x.replace("and", "&") if "and" in x else x,
                lambda x: x + " according to reports" if len(x) < 100 else x
            ]

            mod_func = random.choice(modifications)
            modified = mod_func(base_text)
            result.append(modified)

        return result

def main():
    creator = ImprovedTerrorismDatasetCreator()

    # Create smaller, better-quality dataset
    dataset = creator.create_balanced_dataset(total_size=3000)

    # Save to CSV
    filename = 'terrorism_dataset_improved.csv'
    dataset.to_csv(filename, index=False)

    print(f"Saved as: {filename}")
    print(f"Total samples: {len(dataset)}")
    print("\nLabel distribution:")
    print(dataset['category'].value_counts())
    print("\nText statistics:")
    print(f"Average text length: {dataset['text_length'].mean():.1f}")
    print(f"Average word count: {dataset['word_count'].mean():.1f}")
    print("\nSample texts:")
    for category in ['Safe', 'Suspicious', 'Dangerous']:
        sample = dataset[dataset['category'] == category].iloc[0]
        print(f"\n{category}: {sample['text'][:100]}...")

if __name__ == "__main__":
    main()